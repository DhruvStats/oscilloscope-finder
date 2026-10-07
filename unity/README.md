# Unity / Quest client

`OscilloscopeRecognitionClient.cs` connects a Unity app (Meta Quest) to the recognition server using the
Leonardo AR contract: `GET /health`, `POST /v1/recognitions` (multipart `image`, `session_id`,
`frame_timestamp_ms`), normalised boxes in the answer.

## Use
1. Run the server on the lab PC, reachable on the LAN only:
   `.venv\Scripts\python -m uvicorn server.app:app --host 0.0.0.0 --port 8011`
   (allow port 8011 in the Windows firewall for the private network; never expose it to the Internet).
2. In Unity: install `com.unity.nuget.newtonsoft-json`, copy the script into `Assets/`, add it to a
   GameObject and set **Server Url** to the PC's LAN address, e.g. `http://192.168.1.20:8011`.
3. Camera: assign a `WebCamTexture` (the Quest passthrough camera, through Meta's Passthrough Camera API,
   is exposed as one) or call `SubmitFrame(texture)` from your own capture code.
4. Under **AR modules per instrument**, bind `rs_rtb2004`, `tek_tds2014`, `tek_tds1002` to the AR content
   to show. The script enables the module of each instrument found in the newest frame and hides the others.

## Behaviour
- One request at a time, about 1 frame per second (the CPU server needs 0.3-1 s per frame).
- Answers for an older frame, for another session, or older than 1.5 s are discarded.
- Simulated answers (`simulated: true`) are ignored unless **Accept Simulated** is ticked.
- HTTP 503 means the server has no model weights; there is no fallback, as in the Leonardo notes.
- `ViewportRectOf("tek_tds2014")` gives the instrument's box in viewport coordinates for placing UI.

Note: the hosted Render demo uses the same API, but real use must stay on the local network.
