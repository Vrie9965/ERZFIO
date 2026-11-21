# Every Re:Zero Frame in Order

Big thanks to fearocanity, the person who started the original Every Bocchi The Rock Frame in Order project. At first I was not completely sure why he started it, but I think the concept is awesome. It gives you a really cool way to rewatch a show you love, frame by frame. Best of all, he made it open source so anyone could build one for their own favorite anime. This project is my own spin on his idea.

My first bot project was Onimai. It actually finished the whole anime, but man, it was painful. I had to manually edit and censor thousands of frames by hand in an image editor, which took me days. Almost a year after the run was over, Facebook hit the page with strikes and temporary bans. It was triggered by one single spicy frame I forgot to blur, or maybe from people resharing it. In the end, Facebook nuked the entire page and permanently banned my personal account too. I lost years of personal photos, milestones, and memories. Luckily, my U149 Frames page was on a completely different account, so it survived and is still running today. After that, I started Medalist, which is currently on episode 2.

## Why a new system was needed

Back then, the old method was just using ffmpeg to cut an episode at a fixed frame rate, like 3.5 frames per second. That worked fine at first, but both me and the followers started noticing a huge problem. Anime has tons of amazing keyframes hand-drawn by talented animators that only appear on screen for a split second. A dumb fixed-rate ffmpeg cut blindly skips right past those frames, and losing that art is a big shame.

Because of that, running a raw command like `ffmpeg -hwaccel cuda -i "video.mkv" -r 3.5 -q:v 1 frame_%00d.jpg` just does not cut it anymore. We needed a smarter system. It has to look at the frames and figure out what actually belongs there, detect and censor dangerous content so Facebook does not ban us again, tell characters apart, see what is actually moving versus what is just static, and drop wasted duplicate still frames while keeping camera pans smooth. So I had to build a new system from scratch, and that is what this repo is for.

## How this bot works

The setup itself is pretty clean. The episode gets sliced into numbered pictures (`frame_1.jpg`, `frame_2.jpg`, and so on) and saved in the `frames/` folder. This repo basically acts like a giant git-based photo album.

GitHub Actions acts as our free cloud timer. Every two hours, a GitHub workflow wakes up and runs our bash script, so we do not have to leave a PC running 24/7 or pay for web hosting. During each run, the bot checks `fb/state.json` to see which frame was posted last. Then it posts a small batch of frames straight to the Facebook Page using Facebook Graph API, sleeping for two minutes between each post so Facebook does not freak out and flag us for spam.

The bot never posts duplicate frames. When a batch finishes, it writes the new frame counter back to `fb/state.json` and adds an entry to `fb/log.txt` using a single git commit with `[skip ci]`. If GitHub Actions gets cut off mid-run, the bot knows exactly where it stopped and resumes cleanly. Captions are dynamic templates that auto-fill the anime title, season, episode, frame number, and exact timestamp. For security, our Facebook access token is locked in GitHub encrypted Secrets, so it never gets leaked in public commits. There is also a built-in health check workflow to test your token and frames before going live.
