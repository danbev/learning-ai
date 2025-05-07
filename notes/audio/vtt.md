## Web Video Text Tracks (VTT)
This is a plain text file format for displaying timed text tracks with video
content. The `.vtt` file extension is part of the HTML5 specification and is used
for adding subtitles, captions, descriptions etc to video.


these files sections called `cues`, think of this as in a threater where an actor
would have a `cue` to when it is their turn to speak their lines. The cue has
an optional identifier, then a line with the start and end time of the cue, and
then the text of the cue. And in this case the cue tells the video player to
show the text at the specified time.
```
WEBVTT

00:00:00.000 --> 00:00:11.000
 And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```
