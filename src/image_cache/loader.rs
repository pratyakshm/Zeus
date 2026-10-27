//! Decodes still pictures at the size they are drawn at.
//!
//! egui's own loaders decode a picture at the file's resolution and keep one
//! texture per URI whatever size it is drawn at, so a 640-pixel avatar drawn
//! in a 48-point circle held a 640-pixel texture, and a 12-megapixel photo in
//! a bubble held 46 MB. [`Textures`] keeps one texture per URI too, but at the
//! size its draws ask for: the size hint each draw passes, its rectangle in
//! physical pixels (so HiDPI and zoom are in it), rounded up to a [`bucket`]
//! and never above the picture's own size. A draw that needs more texels
//! than the texture holds has a larger one decoded, and is shown the smaller
//! one until it is ready, so a wider window or a zoom sharpens the pictures
//! instead of blanking them. [`Images`] decodes, on threads of its own, a few
//! at a time.
//!
//! The texture keeps the picture's own size as its `size`, so every layout
//! that reads the size of a loaded picture lays it out as before.
//!
//! SVGs, which egui rasterises at each size, and GIF and WebP files, which it
//! animates frame by frame, are left to egui's loaders.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, PoisonError};

use egui::load::{
    Bytes, BytesPoll, ImageLoadResult, ImageLoader, ImagePoll, LoadError, SizeHint, SizedTexture,
    TextureLoadResult, TextureLoader, TexturePoll,
};
use egui::mutex::Mutex;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Vec2};

/// The smallest side a picture is decoded at.
const SMALLEST_SIDE: u32 = 32;

/// A picture asked for at more than this fraction of its own size is decoded
/// whole: scaling it would save little memory and cost a little sharpness.
const KEEP_SOURCE: f32 = 0.85;

/// How a picture is scaled down: sharper than a triangle filter, and much
/// cheaper than Lanczos on a 12-megapixel photo.
const FILTER: image::imageops::FilterType = image::imageops::FilterType::CatmullRom;

/// Decodes that run at once, across every window. Each holds a whole decoded
/// picture before it is scaled, so a chat of photos scrolled past at once
/// decodes a few at a time rather than all of them together.
const DECODERS: usize = 3;

/// The sizes asked for while a picture decodes.
type Hints = Arc<Mutex<Vec<SizeHint>>>;

/// Draws of one picture in different places ask for at most this many sizes
/// while it decodes; a larger one asked for later is decoded afterwards.
const MOST_HINTS: usize = 16;

/// The texture side a picture whose longer side should be `side` texels is
/// decoded at: 32, 45, 64, 91, 128 and so on, each step √2 larger, so a window
/// resized a little does not decode its pictures again.
pub(super) fn bucket(side: u32) -> u32 {
    let side = u64::from(side);
    let mut even = u64::from(SMALLEST_SIDE);
    loop {
        let odd = (even as f64 * std::f64::consts::SQRT_2).round() as u64;
        for candidate in [even, odd] {
            if candidate >= side {
                return u32::try_from(candidate).unwrap_or(u32::MAX);
            }
        }
        even *= 2;
    }
}

/// The fraction of a `width` × `height` picture's texels that a draw asking
/// for `hint` shows: 1 or more is all of them.
fn scale(hint: SizeHint, [width, height]: [u32; 2]) -> f32 {
    let (width, height) = (width.max(1) as f32, height.max(1) as f32);
    match hint {
        SizeHint::Scale(scale) => scale.0,
        SizeHint::Width(wide) => wide as f32 / width,
        SizeHint::Height(tall) => tall as f32 / height,
        // egui's `Image` fits the picture inside the box, keeping its shape.
        SizeHint::Size {
            width: wide,
            height: tall,
            maintain_aspect_ratio: true,
        } => (wide as f32 / width).min(tall as f32 / height),
        // `paint_at` fills the box, which a UV rectangle may crop the picture
        // to: the picture has to cover it.
        SizeHint::Size {
            width: wide,
            height: tall,
            maintain_aspect_ratio: false,
        } => (wide as f32 / width).max(tall as f32 / height),
    }
}

/// The texel size a `source`-sized picture is decoded at for a draw asking
/// for `hint`: its own size, or smaller with its longer side rounded up to a
/// [`bucket`].
pub(super) fn decode_size(hint: SizeHint, source: [u32; 2]) -> [u32; 2] {
    let long = source[0].max(source[1]);
    let scale = scale(hint, source);
    if long == 0 || scale.is_nan() || scale >= 1.0 {
        return source;
    }
    let wanted = (scale.max(0.0) * long as f32).ceil() as u32;
    let side = bucket(wanted);
    if side as f32 >= long as f32 * KEEP_SOURCE {
        return source;
    }
    let factor = f64::from(side) / f64::from(long);
    let fit = |length: u32| ((f64::from(length) * factor).round() as u32).clamp(1, side);
    [fit(source[0]), fit(source[1])]
}

/// Whether a `texels`-sized texture of a `source`-sized picture serves a draw
/// asking for `hint`: it is as large as a decode for that draw would be.
pub(super) fn serves(texels: [usize; 2], source: [u32; 2], hint: SizeHint) -> bool {
    let wanted = decode_size(hint, source);
    texels[0].max(texels[1]) >= wanted[0].max(wanted[1]) as usize
}

/// A picture's own size in texels, from the size a texture keeps for it.
fn texels_of(size: Vec2) -> [u32; 2] {
    [size.x.round() as u32, size.y.round() as u32]
}

/// Scales a decoded picture down to the largest size `hints` ask for, or
/// leaves it whole, and makes texels of it. The result keeps the picture's
/// own size as its `source_size`.
pub(super) fn scaled(picture: image::DynamicImage, hints: &[SizeHint]) -> ColorImage {
    let source = [picture.width(), picture.height()];
    let [width, height] = hints
        .iter()
        .map(|hint| decode_size(*hint, source))
        .max_by_key(|size| size[0].max(size[1]))
        .unwrap_or(source);
    let size = [width as usize, height as usize];
    if [width, height] == source {
        // Exactly as egui's own loader makes texels of it.
        let rgba = picture.into_rgba8();
        return ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    }
    let small = if picture.color().has_alpha() {
        // Scaled with its alpha premultiplied, as egui's texels are, or the
        // colour of transparent pixels bleeds into the edges as a dark rim.
        let mut rgba = picture.into_rgba8();
        for pixel in rgba.pixels_mut() {
            let [red, green, blue, alpha] = pixel.0;
            pixel.0 = Color32::from_rgba_unmultiplied(red, green, blue, alpha).to_array();
        }
        let mut small = image::imageops::resize(&rgba, width, height, FILTER);
        // The filter overshoots a little at hard edges, and a premultiplied
        // colour cannot be brighter than its alpha.
        for pixel in small.pixels_mut() {
            let alpha = pixel.0[3];
            for channel in &mut pixel.0[..3] {
                *channel = (*channel).min(alpha);
            }
        }
        small
    } else {
        picture.resize_exact(width, height, FILTER).into_rgba8()
    };
    ColorImage::from_rgba_premultiplied(size, small.as_raw())
        .with_source_size(egui::vec2(source[0] as f32, source[1] as f32))
}

/// Decodes `bytes`, turned upright as the file says, at the largest size
/// `hints` holds once the picture is in memory, so a draw that came while it
/// decoded counts.
fn decode(bytes: &[u8], hints: &Hints) -> Result<ColorImage, String> {
    let format = image::guess_format(bytes).map_err(|err| err.to_string())?;
    let mut decoder = image::ImageReader::with_format(std::io::Cursor::new(bytes), format)
        .into_decoder()
        .map_err(|err| err.to_string())?;
    let orientation =
        image::ImageDecoder::orientation(&mut decoder).map_err(|err| err.to_string())?;
    let mut picture = image::DynamicImage::from_decoder(decoder).map_err(|err| err.to_string())?;
    picture.apply_orientation(orientation);
    let asked = hints.lock().clone();
    Ok(scaled(picture, &asked))
}

/// Decodes running now.
static RUNNING: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
/// Signalled when a decode finishes.
static FINISHED: Condvar = Condvar::new();

/// One of the [`DECODERS`] turns to decode, given back when dropped.
struct Turn;

impl Turn {
    fn take() -> Self {
        let mut running = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
        while *running >= DECODERS {
            running = FINISHED
                .wait(running)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *running += 1;
        Self
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut running = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
        *running = running.saturating_sub(1);
        drop(running);
        FINISHED.notify_one();
    }
}

enum Decode {
    /// Waiting for a turn or decoding, with the sizes asked for meanwhile.
    Pending(Hints),
    Ready(Result<Arc<ColorImage>, String>),
}

type Cache = Arc<Mutex<HashMap<String, Decode>>>;

/// Whether `uri`'s entry is still the decode `hints` belongs to: not
/// forgotten, nor forgotten and asked for again, since it started.
fn current(cache: &HashMap<String, Decode>, uri: &str, hints: &Hints) -> bool {
    matches!(cache.get(uri), Some(Decode::Pending(waiting)) if Arc::ptr_eq(waiting, hints))
}

/// Whether a file's name leaves it to the image crate: no extension, or one
/// of a format it reads. Checked before the file is read, as egui's own
/// loader does, so a video or a document is never read just to find out.
fn readable_name(uri: &str) -> bool {
    !uri.starts_with("file://")
        || std::path::Path::new(uri)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| {
                image::ImageFormat::from_extension(extension)
                    .is_some_and(|format| format.reading_enabled())
            })
}

/// Adds a draw's size to a decode under way.
fn ask(hints: &Hints, hint: SizeHint) {
    let mut asked = hints.lock();
    if !asked.contains(&hint) && asked.len() < MOST_HINTS {
        asked.push(hint);
    }
}

/// Decodes still pictures, scaled to the size hints [`Textures`] passes on.
/// It keeps a decoded picture only until [`Textures`] has uploaded it.
#[derive(Default)]
pub(super) struct Images {
    cache: Cache,
}

impl Images {
    pub(super) const ID: &'static str = egui::generate_loader_id!(Images);
}

/// Decodes `bytes` for `uri` on a thread of its own, once a turn is free.
fn start(ctx: &egui::Context, cache: &Cache, uri: &str, bytes: Bytes, hints: Hints) {
    let waker = ctx.clone();
    let shared = Arc::clone(cache);
    let key = uri.to_owned();
    let spawned = std::thread::Builder::new()
        .name("image-decode".to_owned())
        .spawn(move || {
            let _turn = Turn::take();
            // Forgotten while it waited: nothing draws it any more.
            if !current(&shared.lock(), &key, &hints) {
                return;
            }
            let decoded =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(&bytes, &hints)))
                    .unwrap_or_else(|_| Err("the picture could not be decoded".to_owned()))
                    .map(Arc::new);
            let delivered = {
                let mut cache = shared.lock();
                let delivered = current(&cache, &key, &hints);
                if delivered {
                    cache.insert(key, Decode::Ready(decoded));
                }
                delivered
            };
            // Asked with the cache unlocked: the window may be waiting on it
            // while holding the context.
            if delivered {
                waker.request_repaint();
            }
        });
    if spawned.is_err() {
        cache.lock().insert(
            uri.to_owned(),
            Decode::Ready(Err("no thread could decode the picture".to_owned())),
        );
    }
}

impl ImageLoader for Images {
    fn id(&self) -> &str {
        Self::ID
    }

    fn load(&self, ctx: &egui::Context, uri: &str, size_hint: SizeHint) -> ImageLoadResult {
        if !super::still(uri) {
            return Err(LoadError::NotSupported);
        }
        if let Some(known) = self.cache.lock().get(uri) {
            return match known {
                Decode::Pending(hints) => {
                    ask(hints, size_hint);
                    Ok(ImagePoll::Pending { size: None })
                }
                Decode::Ready(Ok(image)) => Ok(ImagePoll::Ready {
                    image: Arc::clone(image),
                }),
                Decode::Ready(Err(err)) => Err(LoadError::Loading(err.clone())),
            };
        }
        let source = super::source_uri(uri);
        if !readable_name(source) {
            return Err(LoadError::NotSupported);
        }
        let bytes = match ctx.try_load_bytes(source) {
            Ok(BytesPoll::Ready { bytes, .. }) => bytes,
            Ok(BytesPoll::Pending { size }) => return Ok(ImagePoll::Pending { size }),
            Err(err) => return Err(err),
        };
        // What the image crate cannot read is left to egui's loaders, as an
        // SVG without its extension is.
        if !image::guess_format(&bytes).is_ok_and(|format| format.reading_enabled()) {
            return Err(LoadError::NotSupported);
        }
        let hints: Hints = Arc::new(Mutex::new(vec![size_hint]));
        self.cache
            .lock()
            .insert(uri.to_owned(), Decode::Pending(Arc::clone(&hints)));
        start(ctx, &self.cache, uri, bytes, hints);
        Ok(ImagePoll::Pending { size: None })
    }

    fn forget(&self, uri: &str) {
        self.cache.lock().remove(uri);
    }

    fn forget_all(&self) {
        self.cache.lock().clear();
    }

    fn byte_size(&self) -> usize {
        self.cache
            .lock()
            .values()
            .map(|decode| match decode {
                Decode::Ready(Ok(image)) => image.pixels.len() * std::mem::size_of::<Color32>(),
                Decode::Ready(Err(err)) => err.len(),
                Decode::Pending(_) => 0,
            })
            .sum()
    }

    fn has_pending(&self) -> bool {
        self.cache
            .lock()
            .values()
            .any(|decode| matches!(decode, Decode::Pending(_)))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    uri: String,
    options: TextureOptions,
}

struct Slot {
    handle: TextureHandle,
    /// The picture's own size, which layouts read.
    source: Vec2,
    /// The loaders gave back no larger picture than this one when asked for
    /// one, so asking again would decode the same picture on every frame.
    limit: bool,
}

impl Slot {
    fn texture(&self) -> SizedTexture {
        SizedTexture::new(self.handle.id(), self.source)
    }

    fn longest(&self) -> usize {
        let [width, height] = self.handle.size();
        width.max(height)
    }

    fn serves(&self, hint: SizeHint) -> bool {
        self.limit || serves(self.handle.size(), texels_of(self.source), hint)
    }
}

/// Uploads still pictures at the size their draws ask for; see the module.
#[derive(Default)]
pub(super) struct Textures {
    slots: Mutex<HashMap<Key, Slot>>,
}

impl Textures {
    pub(super) const ID: &'static str = egui::generate_loader_id!(Textures);
}

/// The texture of the picture a full-size `key` is the preview of, shown
/// while the full-size one decodes, and kept resident while it stands in.
fn stand_in(ctx: &egui::Context, slots: &HashMap<Key, Slot>, key: &Key) -> Option<SizedTexture> {
    let base = super::base_uri(&key.uri)?;
    let slot = slots.get(&Key {
        uri: base.to_owned(),
        options: key.options,
    })?;
    super::touch(ctx, base);
    Some(slot.texture())
}

/// Drops the copies a picture's upload no longer needs, as egui's
/// `reduce_texture_memory` does: the decoded texels and the file's bytes.
/// The bytes of a `bytes://` picture stay: they are its message's small
/// embedded thumbnail, registered only once, and a larger draw decodes them
/// again. They go with the rest when the image cache forgets the picture.
fn release(ctx: &egui::Context, uri: &str) {
    let loaders = ctx.loaders();
    let source = super::source_uri(uri);
    if !source.starts_with("bytes://") {
        for loader in loaders.bytes.lock().iter() {
            loader.forget(source);
        }
    }
    for loader in loaders.image.lock().iter() {
        loader.forget(uri);
    }
}

impl TextureLoader for Textures {
    fn id(&self) -> &str {
        Self::ID
    }

    fn load(
        &self,
        ctx: &egui::Context,
        uri: &str,
        texture_options: TextureOptions,
        size_hint: SizeHint,
    ) -> TextureLoadResult {
        if !super::still(uri) {
            return Err(LoadError::NotSupported);
        }
        // Every picture these loaders hold is in the image cache's resident
        // window, whichever draw site asked for it.
        super::touch(ctx, uri);
        // The preview's own texture is the whole picture, so zooming in
        // shows its every pixel at once.
        let hint = if super::is_full_size(uri) {
            SizeHint::default()
        } else {
            size_hint
        };
        let key = Key {
            uri: uri.to_owned(),
            options: texture_options,
        };
        let mut slots = self.slots.lock();
        let held = slots
            .get(&key)
            .map(|slot| (slot.texture(), slot.serves(hint)));
        if let Some((texture, true)) = held {
            return Ok(TexturePoll::Ready { texture });
        }
        let held = held.map(|(texture, _)| texture);
        match ctx.try_load_image(uri, hint) {
            Ok(ImagePoll::Ready { image }) => {
                let longest = image.size[0].max(image.size[1]);
                let unchanged = slots
                    .get_mut(&key)
                    .filter(|slot| slot.longest() >= longest)
                    .map(|slot| {
                        slot.limit = true;
                        slot.texture()
                    });
                if let Some(texture) = unchanged {
                    drop(slots);
                    release(ctx, uri);
                    return Ok(TexturePoll::Ready { texture });
                }
                let source = image.source_size;
                let handle = ctx.load_texture(uri, image, texture_options);
                super::uploaded(ctx, uri, handle.byte_size());
                let slot = Slot {
                    handle,
                    source,
                    limit: false,
                };
                let texture = slot.texture();
                // Replacing a smaller texture frees it once this frame is
                // painted, so a draw earlier in the frame still shows it.
                slots.insert(key, slot);
                drop(slots);
                release(ctx, uri);
                Ok(TexturePoll::Ready { texture })
            }
            Ok(ImagePoll::Pending { size }) => {
                Ok(match held.or_else(|| stand_in(ctx, &slots, &key)) {
                    Some(texture) => TexturePoll::Ready { texture },
                    None => TexturePoll::Pending { size },
                })
            }
            // A larger decode that fails leaves the smaller picture shown.
            Err(err) => match held.or_else(|| stand_in(ctx, &slots, &key)) {
                Some(texture) => Ok(TexturePoll::Ready { texture }),
                None => Err(err),
            },
        }
    }

    fn forget(&self, uri: &str) {
        self.slots.lock().retain(|key, _| key.uri != uri);
    }

    fn forget_all(&self) {
        self.slots.lock().clear();
    }

    fn byte_size(&self) -> usize {
        self.slots
            .lock()
            .values()
            .map(|slot| slot.handle.byte_size())
            .sum()
    }
}
