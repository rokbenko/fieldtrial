# Testing the console on a phone

A short manual check on real devices, before relying on the console in a study. It takes
about 15 minutes. You need a laptop and a phone on the same Wi-Fi network.

## Setup

1. On the laptop: `fieldtrial demo --dir ~/fieldtrial-demo --no-browser`. Note the
   folder; press Ctrl+C.
2. Serve it on the network: `fieldtrial serve ~/fieldtrial-demo --lan`.
3. Scan the printed QR code with the phone's camera.

## Checks

- [ ] The study page opens on the phone, and the address bar no longer shows the token.
- [ ] Opening the URL without the token (in a private tab) shows the access-token message.
- [ ] Start a session: the browser refuses to submit until every rig check is ticked.
- [ ] The blind code is large and readable at arm's length; the arm name is not shown.
- [ ] Start, wait 10 seconds, Stop: the timer counts smoothly and stops; the stop time shown
      matches.
- [ ] The simulated runner's suggestion is pre-selected; change the stage and confirm.
- [ ] Undo within 10 seconds brings back the label form; after 10 seconds the Undo button
      disappears.
- [ ] Double-tap Confirm quickly: only one trial is recorded (check Trial history).
- [ ] Mark a trial invalid with a reason: the next trial shown is its replacement.
- [ ] Open the mirror on the laptop (Open a mirror screen): starting and stopping on the
      phone updates the laptop within a second.
- [ ] Turn the phone sideways and back: nothing overlaps and no text is cut off.
- [ ] Switch the phone to dark mode: text and buttons stay high-contrast.
- [ ] Lock the phone for a minute mid-trial, unlock: the timer shows the right elapsed time.
- [ ] Turn Wi-Fi off, tap Confirm, turn it back on and tap again: the trial is recorded once.
- [ ] With a Bluetooth or USB keyboard (or foot pedal): Space starts and stops, 1–4 pick a
      stage, Enter confirms; rebind Start/Stop to another key and check it works.
- [ ] Trial history: correct one trial with a reason; the row updates.
- [ ] Report page: unblind (tick the box), and the per-arm table appears.

Report anything that fails as an issue, with the phone model and browser.
