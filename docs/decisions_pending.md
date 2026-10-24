## PENDING
- Album covers and song previews for the demo timeline from the Discogs REST API (the bulk dump has no image or video fields; the API gives images[].uri and videos[].uri per release, only the demo artists would need it, limits 60 requests/min unauthenticated and 240 with a free token, embed the YouTube link rather than hosting audio). Not implemented.
- Modeling extras: a stacking ensemble (meta-model on out-of-fold predictions of the best models) and PCA as preprocessing for KNN if the final WHAT feature set is high-dimensional. Not done.
