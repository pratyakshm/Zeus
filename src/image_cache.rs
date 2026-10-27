//! Bounds what the images Zeus draws hold.
//!
//! egui caches the bytes, the decoded pixels, and the GPU texture of every URI
//! it has loaded. It only ever evicts the textures of `.svg` files, so left
//! alone a session that scrolls through photos holds every one of them for
//! good. Each draw site registers its URI here, and so does the loader that
//! decodes still pictures at the size they are drawn at ([`install`]).
//! [`sweep`] forgets the images nothing has drawn for a minute, and the ones
//! that fall out of the resident window, which holds 96 images and 64 MiB of
//! textures at most besides the ones on screen.

mod loader;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Image URIs kept resident. Anything the current frame drew is always kept,
/// so this only bounds what has scrolled away.
const RESIDENT: usize = 96;

/// Bytes the textures of images that are not on screen may hold, counted for
/// the pictures Zeus's loader decodes. A frame that shows more keeps it all.
const BUDGET: usize = 64 * 1024 * 1024;

/// Seconds an image nothing draws stays resident: scrolling back within a
/// minute finds it loaded, and later it loads again as an image never shown
/// does.
const IDLE_SECONDS: f64 = 60.0;

/// Full-size pictures the open preview keeps besides the one it shows, so
/// stepping back to the previous one is instant.
const FULL_SIZE_KEPT: usize = 1;

/// Marks the URI of a picture's own full-size texture; see [`full_size_uri`].
const FULL_SIZE: &str = "#zeus-full-size";

struct Entry {
    /// Frame the image was last drawn on.
    frame: u64,
    /// When it was last drawn, in egui's seconds.
    seen: f64,
    /// Whether its bytes are registered with egui's byte loader. egui's own
    /// loaders drop them once the texture is uploaded, Zeus's keep those of a
    /// still picture until it is forgotten; a forgotten image's next draw
    /// registers them again.
    registered: bool,
    /// Bytes its texture holds, once Zeus's loader uploaded one.
    texture: usize,
    /// Bytes registered for it.
    included: usize,
}

impl Entry {
    fn new(frame: u64, seen: f64) -> Self {
        Self {
            frame,
            seen,
            registered: false,
            texture: 0,
            included: 0,
        }
    }

    fn cost(&self) -> usize {
        self.texture.saturating_add(self.included)
    }
}

#[derive(Clone, Default)]
struct Registry(Arc<Mutex<HashMap<String, Entry>>>);

fn registry_id() -> egui::Id {
    egui::Id::new("image-cache")
}

fn registry(ctx: &egui::Context) -> Registry {
    ctx.data_mut(|data| {
        data.get_temp_mut_or_default::<Registry>(registry_id())
            .clone()
    })
}

/// The frame being drawn and its time.
fn moment(ctx: &egui::Context) -> (u64, f64) {
    (ctx.cumulative_frame_nr(), ctx.input(|input| input.time))
}

/// Installs the loaders that decode still pictures at the size they are
/// drawn at, rounded up to a few steps and never above the picture's own
/// size, so an avatar in a small circle holds a small texture. SVGs, GIFs and
/// WebP files stay with egui's loaders. Call it after egui_extras installs
/// its loaders, so these are asked first.
pub fn install(ctx: &egui::Context) {
    if !ctx.is_loader_installed(loader::Images::ID) {
        ctx.add_image_loader(Arc::new(loader::Images::default()));
    }
    if !ctx.is_loader_installed(loader::Textures::ID) {
        ctx.add_texture_loader(Arc::new(loader::Textures::default()));
    }
}

/// Whether `uri` is a still picture the loaders [`install`] adds decode at its
/// drawn size. SVGs, which egui rasterises at each size, and GIF and WebP
/// files, which it animates (their URIs gain a frame number when drawn), are
/// egui's.
fn still(uri: &str) -> bool {
    !["svg", "gif", "webp"]
        .iter()
        .any(|extension| egui::load::has_extension(uri, extension))
        && egui::decode_animated_image_uri(uri).is_err()
}

/// The URI of `uri`'s full-resolution texture, for the image preview. It is
/// kept apart from the texture the picture's message draws at bubble size,
/// so opening the preview does not leave the bubble holding the whole
/// picture, and it is released once the preview stops drawing it. Until it
/// is decoded the preview is shown the bubble's texture. Animations and SVGs
/// keep their own URI: egui's loaders draw those.
pub fn full_size_uri(uri: &str) -> String {
    if still(uri) {
        format!("{uri}{FULL_SIZE}")
    } else {
        uri.to_owned()
    }
}

fn is_full_size(uri: &str) -> bool {
    uri.ends_with(FULL_SIZE)
}

/// The picture a full-size URI is the preview of.
fn base_uri(uri: &str) -> Option<&str> {
    uri.strip_suffix(FULL_SIZE)
}

/// The URI whose bytes `uri` is decoded from.
fn source_uri(uri: &str) -> &str {
    base_uri(uri).unwrap_or(uri)
}

/// How many image URIs the resident window is tracking. Zero when nothing
/// has been registered, and this does not create the registry to find out.
pub fn len(ctx: &egui::Context) -> usize {
    let Some(registry) = ctx.data(|data| data.get_temp::<Registry>(registry_id())) else {
        return 0;
    };
    registry
        .0
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len()
}

/// Registers `bytes` under `uri`, and marks the image as drawn this frame.
///
/// Re-registers them after [`sweep`] has forgotten the image, or after egui
/// has dropped them once the texture was uploaded.
pub fn include(ctx: &egui::Context, uri: &str, bytes: &[u8]) {
    let (frame, seen) = moment(ctx);
    let registry = registry(ctx);
    let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
    let fresh = match entries.get_mut(uri) {
        Some(entry) => {
            entry.frame = frame;
            entry.seen = seen;
            let fresh = !std::mem::replace(&mut entry.registered, true);
            if fresh {
                entry.included = bytes.len();
            }
            fresh
        }
        None => {
            entries.insert(
                uri.to_owned(),
                Entry {
                    registered: true,
                    included: bytes.len(),
                    ..Entry::new(frame, seen)
                },
            );
            true
        }
    };
    drop(entries);
    if fresh {
        ctx.include_bytes(uri.to_owned(), bytes.to_vec());
    }
}

/// Marks an image URI as drawn this frame.
pub fn touch(ctx: &egui::Context, uri: &str) {
    let (frame, seen) = moment(ctx);
    let registry = registry(ctx);
    let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
    match entries.get_mut(uri) {
        Some(entry) => {
            entry.frame = frame;
            entry.seen = seen;
        }
        None => {
            entries.insert(uri.to_owned(), Entry::new(frame, seen));
        }
    }
}

/// Records the bytes `uri`'s texture now holds, for the byte budget.
fn uploaded(ctx: &egui::Context, uri: &str, bytes: usize) {
    let registry = registry(ctx);
    let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(entry) = entries.get_mut(uri) {
        entry.texture = bytes;
    }
}

/// Forgets one image now rather than once it falls out of the resident
/// window: the info panel frees its thumbnails when it closes. A later draw
/// registers the bytes again. The picture's full-size texture goes with it.
pub fn forget(ctx: &egui::Context, uri: &str) {
    let full = full_size_uri(uri);
    {
        let registry = registry(ctx);
        let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
        entries.remove(uri);
        entries.remove(&full);
    }
    ctx.forget_image(uri);
    if full != uri {
        ctx.forget_image(&full);
    }
}

/// Forgets the images nothing has drawn for a minute, the preview's
/// full-size pictures once it stops drawing them, and then the images that
/// have not been drawn for the longest, until the rest fit the resident
/// window.
///
/// Images the current frame drew are never forgotten, so a frame that shows
/// more than the window holds does not reload them on the next one. It never
/// asks for a frame: images not on screen go on the next frame that runs.
pub fn sweep(ctx: &egui::Context) {
    let (frame, now) = moment(ctx);
    let stale = {
        let registry = registry(ctx);
        let mut entries = registry.0.lock().unwrap_or_else(|p| p.into_inner());
        let stale = stale(&entries, frame, now);
        for uri in &stale {
            entries.remove(uri);
        }
        stale
    };
    for uri in stale {
        ctx.forget_image(&uri);
    }
}

/// Whether the current frame drew a full-size picture: the preview is open.
fn previewing(entries: &HashMap<String, Entry>, frame: u64) -> bool {
    entries
        .iter()
        .any(|(uri, entry)| entry.frame >= frame && is_full_size(uri))
}

/// The URIs a sweep on `frame` at `now` forgets, oldest first; see [`sweep`].
fn stale(entries: &HashMap<String, Entry>, frame: u64, now: f64) -> Vec<String> {
    let previewing = previewing(entries, frame);
    let mut idle: Vec<(&str, &Entry)> = entries
        .iter()
        .filter(|(_, entry)| entry.frame < frame)
        .map(|(uri, entry)| (uri.as_str(), entry))
        .collect();
    idle.sort_unstable_by(|(a, first), (b, second)| {
        first.frame.cmp(&second.frame).then_with(|| a.cmp(b))
    });
    let mut gone: Vec<bool> = idle
        .iter()
        .map(|(_, entry)| now - entry.seen >= IDLE_SECONDS)
        .collect();
    // The preview's pictures, newest first: while it is open it keeps the
    // previous one too, and once it closes they go after a frame's grace, in
    // case a frame did not draw it.
    let mut kept = 0;
    for (index, (uri, entry)) in idle.iter().enumerate().rev() {
        if !is_full_size(uri) {
            continue;
        }
        let keep = if previewing {
            kept += 1;
            kept <= FULL_SIZE_KEPT
        } else {
            frame - entry.frame <= 1
        };
        if !keep {
            gone[index] = true;
        }
    }
    let mut count = entries.len();
    let mut bytes: usize = entries.values().map(Entry::cost).sum();
    for (index, (_, entry)) in idle.iter().enumerate() {
        if gone[index] {
            count -= 1;
            bytes = bytes.saturating_sub(entry.cost());
        }
    }
    for (index, (_, entry)) in idle.iter().enumerate() {
        if count <= RESIDENT && bytes <= BUDGET {
            break;
        }
        if !gone[index] {
            gone[index] = true;
            count -= 1;
            bytes = bytes.saturating_sub(entry.cost());
        }
    }
    idle.iter()
        .zip(gone)
        .filter(|(_, gone)| *gone)
        .map(|((uri, _), _)| (*uri).to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> egui::Context {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        ctx
    }

    /// Advances the frame counter the way a repaint would.
    fn frame(ctx: &egui::Context) {
        let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
    }

    fn registered(ctx: &egui::Context, uri: &str) -> bool {
        ctx.try_load_bytes(uri).is_ok()
    }

    fn uri(index: usize) -> String {
        format!("bytes://test-{index:04}")
    }

    /// Answers in the same frame, so a test can watch a texture being
    /// allocated without waiting on a decoding thread.
    struct ReadyLoader;

    impl egui::load::ImageLoader for ReadyLoader {
        fn id(&self) -> &str {
            "zeus::image_cache::tests::ReadyLoader"
        }

        fn load(
            &self,
            _ctx: &egui::Context,
            _uri: &str,
            _size_hint: egui::load::SizeHint,
        ) -> egui::load::ImageLoadResult {
            Ok(egui::load::ImagePoll::Ready {
                image: std::sync::Arc::new(egui::ColorImage::example()),
            })
        }

        fn forget(&self, _uri: &str) {}

        fn forget_all(&self) {}

        fn byte_size(&self) -> usize {
            0
        }
    }

    /// Draws `shown` and then sweeps, the way a frame of the app does.
    fn draw(ctx: &egui::Context, shown: &[String]) -> egui::FullOutput {
        ctx.run_ui(egui::RawInput::default(), |ui| {
            for uri in shown {
                ui.add(egui::Image::from_uri(uri.clone()));
            }
            sweep(ui.ctx());
        })
    }

    fn allocated(ctx: &egui::Context) -> usize {
        let manager = ctx.tex_manager();
        manager.read().num_allocated()
    }

    #[test]
    fn a_forgotten_image_is_released_and_registers_again_when_drawn() {
        let ctx = context();
        include(&ctx, &uri(1), b"image bytes");
        assert!(registered(&ctx, &uri(1)));
        forget(&ctx, &uri(1));
        assert!(
            !registered(&ctx, &uri(1)),
            "forgotten at once, not swept later"
        );
        include(&ctx, &uri(1), b"image bytes");
        assert!(registered(&ctx, &uri(1)), "a later draw registers it again");
    }

    #[test]
    fn images_the_current_frame_drew_are_never_released() {
        let ctx = context();
        let total = RESIDENT + 40;
        for index in 0..total {
            include(&ctx, &uri(index), b"image bytes");
        }
        // A frame that shows more images than the window holds must not drop
        // the ones it is showing right now.
        sweep(&ctx);
        for index in 0..total {
            assert!(registered(&ctx, &uri(index)), "{} was released", uri(index));
        }
    }

    #[test]
    fn images_that_scroll_away_are_released() {
        let ctx = context();
        let total = RESIDENT + 40;
        for index in 0..total {
            include(&ctx, &uri(index), b"image bytes");
        }
        frame(&ctx);
        sweep(&ctx);
        let kept = (0..total)
            .filter(|index| registered(&ctx, &uri(*index)))
            .count();
        assert_eq!(kept, RESIDENT, "the window should hold exactly its budget");
    }

    #[test]
    fn a_released_image_is_registered_again_when_it_comes_back() {
        let ctx = context();
        for index in 0..RESIDENT + 40 {
            include(&ctx, &uri(index), b"image bytes");
        }
        frame(&ctx);
        sweep(&ctx);
        // The oldest URI lost its bytes. Drawing it again must put them back,
        // because egui has no other way to resolve a `bytes://` image.
        include(&ctx, &uri(0), b"image bytes");
        assert!(registered(&ctx, &uri(0)));
    }

    #[test]
    fn images_that_scroll_away_free_their_texture() {
        let ctx = context();
        ctx.add_image_loader(std::sync::Arc::new(ReadyLoader));
        let uris: Vec<String> = (0..RESIDENT + 8).map(uri).collect();
        for uri in &uris {
            include(&ctx, uri, b"image bytes");
        }
        let mut showing = draw(&ctx, &uris);
        let shown = allocated(&ctx);
        assert!(
            shown >= uris.len(),
            "a drawn image should own a texture, saw {shown} for {} images",
            uris.len()
        );
        showing.textures_delta.clear();

        // Nothing on screen: the images past the window go, and the frame that
        // released them is the one that tells the painter to delete them.
        let mut empty = draw(&ctx, &[]);
        assert_eq!(
            empty.textures_delta.free.len(),
            8,
            "the swept textures should be freed by the painter"
        );
        assert_eq!(allocated(&ctx), shown - 8);
        empty.textures_delta.clear();
    }

    fn size(width: u32, height: u32, fit: bool) -> egui::load::SizeHint {
        egui::load::SizeHint::Size {
            width,
            height,
            maintain_aspect_ratio: fit,
        }
    }

    #[test]
    fn sizes_round_up_in_steps_of_the_square_root_of_two() {
        assert_eq!(loader::bucket(0), 32);
        assert_eq!(loader::bucket(32), 32);
        assert_eq!(loader::bucket(33), 45);
        assert_eq!(loader::bucket(46), 64);
        assert_eq!(loader::bucket(96), 128);
        assert_eq!(loader::bucket(660), 724);
        assert_eq!(loader::bucket(725), 1024);
        let mut last = 0;
        for side in 1..5_000 {
            let bucket = loader::bucket(side);
            assert!(bucket >= side, "{side} rounds down to {bucket}");
            assert!(
                f64::from(bucket) <= f64::from(side.max(32)) * std::f64::consts::SQRT_2 + 1.0,
                "{side} rounds up too far, to {bucket}"
            );
            assert!(bucket >= last, "{side} rounds below a smaller side");
            last = bucket;
        }
    }

    #[test]
    fn a_picture_is_decoded_at_the_size_it_is_drawn_at() {
        // A 640-pixel avatar in a 48-point circle at twice the density.
        assert_eq!(
            loader::decode_size(size(96, 96, true), [640, 640]),
            [128, 128]
        );
        // A 12-megapixel photo in a 330-point bubble, at 2x and at 1x.
        assert_eq!(
            loader::decode_size(size(660, 495, true), [4032, 3024]),
            [724, 543]
        );
        let single = loader::decode_size(size(330, 248, true), [4032, 3024]);
        assert!(single[0] >= 330 && single[0] < 660, "{single:?}");
        // Never above the picture's own size, nor scaled for a sliver of it.
        assert_eq!(
            loader::decode_size(size(3000, 3000, true), [1600, 1200]),
            [1600, 1200]
        );
        assert_eq!(
            loader::decode_size(size(1300, 975, true), [1600, 1200]),
            [1600, 1200]
        );
        assert_eq!(
            loader::decode_size(egui::load::SizeHint::default(), [4032, 3024]),
            [4032, 3024]
        );
    }

    #[test]
    fn a_cropped_picture_is_decoded_to_cover_its_box() {
        // A panorama in a square: fitted inside, it needs the square's width
        // across; cropped to cover it, the square's height down.
        assert_eq!(
            loader::decode_size(size(200, 200, true), [4000, 1000]),
            [256, 64]
        );
        assert_eq!(
            loader::decode_size(size(200, 200, false), [4000, 1000]),
            [1024, 256]
        );
    }

    #[test]
    fn a_texture_serves_draws_up_to_its_size() {
        let source = [4032, 3024];
        let texels = [724, 543];
        assert!(loader::serves(texels, source, size(660, 495, true)));
        assert!(loader::serves(texels, source, size(96, 96, true)));
        assert!(!loader::serves(texels, source, size(1000, 750, true)));
        assert!(!loader::serves(
            texels,
            source,
            egui::load::SizeHint::default()
        ));
        assert!(loader::serves(
            [4032, 3024],
            source,
            egui::load::SizeHint::default()
        ));
    }

    #[test]
    fn a_transparent_picture_scales_without_a_dark_rim() {
        let mut picture = image::RgbaImage::new(64, 64);
        for (x, _, pixel) in picture.enumerate_pixels_mut() {
            *pixel = if x < 32 {
                image::Rgba([255, 255, 255, 255])
            } else {
                image::Rgba([0, 0, 0, 0])
            };
        }
        let scaled = loader::scaled(
            image::DynamicImage::ImageRgba8(picture),
            &[size(16, 16, true)],
        );
        assert_eq!(scaled.size, [32, 32]);
        assert_eq!(scaled.source_size, egui::vec2(64.0, 64.0));
        // White at every opacity: premultiplied, its colour equals its alpha.
        // Scaled with straight alpha, the edge would darken towards black.
        for pixel in &scaled.pixels {
            let [red, green, blue, alpha] = pixel.to_array();
            assert_eq!([red, green, blue], [alpha; 3], "{pixel:?}");
        }
        assert!(scaled.pixels.iter().any(|pixel| pixel.a() == 255));
        assert!(scaled.pixels.iter().any(|pixel| pixel.a() == 0));
    }

    /// Asks for `uri`'s texture until it is ready with texels `done` accepts.
    fn texture_until(
        ctx: &egui::Context,
        uri: &str,
        hint: egui::load::SizeHint,
        done: impl Fn([usize; 2]) -> bool,
    ) -> egui::load::SizedTexture {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Ok(egui::load::TexturePoll::Ready { texture }) =
                ctx.try_load_texture(uri, egui::TextureOptions::default(), hint)
            {
                let texels = ctx
                    .tex_manager()
                    .read()
                    .meta(texture.id)
                    .map(|meta| meta.size);
                if texels.is_some_and(&done) {
                    return texture;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{uri} did not load as expected"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn shown(
        ctx: &egui::Context,
        uri: &str,
        hint: egui::load::SizeHint,
    ) -> Option<egui::TextureId> {
        match ctx.try_load_texture(uri, egui::TextureOptions::default(), hint) {
            Ok(egui::load::TexturePoll::Ready { texture }) => Some(texture.id),
            _ => None,
        }
    }

    /// A 1200×900 photo, registered under `uri`.
    fn photo(ctx: &egui::Context, uri: &str) {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            1200,
            900,
            image::Rgb([200, 120, 40]),
        ))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
        include(ctx, uri, &png);
    }

    #[test]
    fn a_large_picture_gets_a_texture_no_larger_than_its_draw() {
        let ctx = context();
        install(&ctx);
        let uri = "bytes://large-photo";
        photo(&ctx, uri);
        let small = texture_until(&ctx, uri, size(120, 90, true), |texels| texels == [128, 96]);
        assert_eq!(
            small.size,
            egui::vec2(1200.0, 900.0),
            "layouts still read the picture's own size"
        );
        // A larger draw is shown the small texture until its own is ready.
        assert_eq!(
            shown(&ctx, uri, egui::load::SizeHint::default()),
            Some(small.id)
        );
        let whole = texture_until(&ctx, uri, egui::load::SizeHint::default(), |texels| {
            texels == [1200, 900]
        });
        assert_ne!(whole.id, small.id);
        // And a small draw is then served by the larger texture.
        assert_eq!(shown(&ctx, uri, size(120, 90, true)), Some(whole.id));
    }

    #[test]
    fn the_preview_stands_in_the_bubbles_texture_until_its_own_is_decoded() {
        let ctx = context();
        install(&ctx);
        let uri = "bytes://previewed-photo";
        photo(&ctx, uri);
        let bubble = texture_until(&ctx, uri, size(120, 90, true), |texels| texels == [128, 96]);
        let full = full_size_uri(uri);
        assert_ne!(full, uri);
        assert_eq!(shown(&ctx, &full, size(120, 90, true)), Some(bubble.id));
        // Its own texture is the whole picture, whatever it is drawn at.
        let own = texture_until(&ctx, &full, size(120, 90, true), |texels| {
            texels == [1200, 900]
        });
        assert_ne!(own.id, bubble.id);
        assert_eq!(
            shown(&ctx, uri, size(120, 90, true)),
            Some(bubble.id),
            "the bubble keeps its small texture"
        );
    }

    #[test]
    fn animations_and_svgs_keep_their_uri_for_the_preview() {
        for uri in [
            "file:///stickers/a.webp",
            "file:///gifs/a.gif",
            "bytes://icon.svg",
        ] {
            assert_eq!(full_size_uri(uri), uri);
        }
        assert_ne!(
            full_size_uri("file:///photos/a.jpg"),
            "file:///photos/a.jpg"
        );
    }

    /// Runs a frame at `time` seconds that calls `draw`.
    fn pass_at(ctx: &egui::Context, time: f64, mut draw: impl FnMut(&egui::Context)) {
        let input = egui::RawInput {
            time: Some(time),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| draw(ui.ctx()));
        output.textures_delta.clear();
    }

    #[test]
    fn images_nothing_draws_for_a_minute_are_released() {
        let ctx = context();
        let (kept, left) = (uri(1), uri(2));
        pass_at(&ctx, 0.0, |ctx| {
            include(ctx, &kept, b"image bytes");
            include(ctx, &left, b"image bytes");
            sweep(ctx);
        });
        let on_screen = |ctx: &egui::Context| {
            include(ctx, &kept, b"image bytes");
            sweep(ctx);
        };
        pass_at(&ctx, 30.0, on_screen);
        assert!(registered(&ctx, &left), "half a minute away it stays");
        pass_at(&ctx, 61.0, on_screen);
        assert!(!registered(&ctx, &left), "a minute away it is released");
        assert!(registered(&ctx, &kept), "the image on screen stays");
    }

    fn entry(frame: u64, seen: f64, texture: usize) -> Entry {
        Entry {
            texture,
            registered: true,
            ..Entry::new(frame, seen)
        }
    }

    #[test]
    fn the_oldest_pictures_go_once_the_window_holds_more_than_its_budget() {
        // Five 20 MiB pictures drawn a frame apart, the newest on screen.
        let mut entries: HashMap<String, Entry> = (0..5)
            .map(|index| (uri(index), entry(index as u64, 0.0, 20 << 20)))
            .collect();
        assert_eq!(stale(&entries, 4, 1.0), vec![uri(0), uri(1)]);
        // A frame that draws more than the budget keeps all of it.
        for entry in entries.values_mut() {
            entry.frame = 4;
        }
        assert!(stale(&entries, 4, 1.0).is_empty());
    }

    #[test]
    fn the_previews_full_size_pictures_go_when_it_closes() {
        let (shown, previous, older) = (
            full_size_uri(&uri(1)),
            full_size_uri(&uri(2)),
            full_size_uri(&uri(3)),
        );
        let mut entries = HashMap::from([
            (uri(1), entry(10, 0.0, 1)),
            (shown.clone(), entry(10, 0.0, 1)),
            (previous.clone(), entry(9, 0.0, 1)),
            (older.clone(), entry(8, 0.0, 1)),
        ]);
        // Open: it keeps the picture shown a step ago, not the one before.
        assert_eq!(stale(&entries, 10, 0.0), vec![older.clone()]);
        entries.remove(&older);
        // Closed: a frame's grace, then both go.
        assert_eq!(stale(&entries, 11, 0.0), vec![previous.clone()]);
        assert_eq!(stale(&entries, 12, 0.0), vec![previous, shown]);
    }
}
