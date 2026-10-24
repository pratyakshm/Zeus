# Todo

- Sending a message while the chat is at the bottom still jumps the transcript up instantly. The outgoing bubble springs in. Commit `2c4aab8` was meant to ease that shift: the glide uses the same 24-point tail slack as the rest of the transcript, discards a pass so the shown frame is already at the new offset, and slides panel-clipped text. The headless test `a_new_message_eases_the_open_transcript_up` passes. Tried in the live app on 2026-10-25 and the list still snaps.
