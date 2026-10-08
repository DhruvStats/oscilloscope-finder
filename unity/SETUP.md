# Option A - Quest / Unity with the lab server (step by step)

```
Quest camera -> Wi-Fi -> lab PC  POST /v1/recognitions -> name + box -> Unity shows the AR module
```

## 1. Lab PC: start the server
```powershell
cd C:\Users\WAGH\oscilloscope-detection
.\tools\start_lan_server.ps1
```
It prints the addresses to use, e.g. `http://192.168.137.1:8011`.

**Network.** Campus Wi-Fi (eduroam) usually blocks device-to-device traffic, so the Quest cannot reach
the PC there. Use the PC's own hotspot instead:
- Windows Settings > Network & internet > **Mobile hotspot** > On (note name and password).
- Connect the Quest to that hotspot. The PC is then always `192.168.137.1`.

**Firewall (one-time, by you, as administrator).** Allow port 8011 on private networks only:
```powershell
New-NetFirewallRule -DisplayName "Oscilloscope finder 8011" -Direction Inbound -Protocol TCP -LocalPort 8011 -Profile Private -Action Allow
```
**Check:** open `http://192.168.137.1:8011/health` in the Quest browser -> you should see `"status":"ok"`.

## 2. Unity project
- Unity 2022.3 LTS or newer, Android build target, **Meta XR SDK** (All-in-One) installed.
- Package Manager > Add by name: `com.unity.nuget.newtonsoft-json`.
- Copy `OscilloscopeRecognitionClient.cs`, `CameraStarter.cs`, `RecognitionOverlay.cs` into `Assets/Scripts/`.
- Player Settings > Other: **Internet Access = Require**; allow plain HTTP (Unity 2022+:
  *Allow downloads over HTTP = Always allowed*) because the lab server uses `http://`.

## 3. Scene
1. Empty GameObject **Recognition** with
   - `OscilloscopeRecognitionClient`: Server Url = `http://192.168.137.1:8011`, Send Rate 1
   - `CameraStarter`: Client = the component above
   - `RecognitionOverlay`: Client = the same (debug boxes on screen)
2. Under **AR modules per instrument** add three entries: `rs_rtb2004`, `tek_tds2014`, `tek_tds1002`,
   each linked to the GameObject with that instrument's AR content (start them disabled).

**Editor test first:** press Play - the PC webcam is used. Point it at an oscilloscope or a photo of one
on another screen; boxes appear and the right AR module switches on.

## 4. Quest camera (Quest 3 / 3S)
The headset camera is available to apps through Meta's **Passthrough Camera API** (Horizon OS v74 or newer):
- Android manifest permissions: `android.permission.CAMERA` and `horizonos.permission.HEADSET_CAMERA`
  (CameraStarter asks for both at start-up).
- The passthrough camera then appears as a `WebCamTexture`; set CameraStarter's *Device Name Contains*
  if several are listed (left/right camera).
- Follow Meta's current "Passthrough Camera API" samples for the exact SDK settings - they change between
  SDK versions.

## 5. Using the result in AR
- `client.Latest.detections` - every instrument found: `label`, `display_name`, `confidence`, `bbox` (0-1).
- `client.ViewportRectOf("tek_tds2014")` - where it is on screen, to place a label or panel.
- `onRecognition` event - hook your own logic (e.g. open the right manual page).
- Stale answers (older frame, other session, older than 1.5 s) and simulated answers are ignored.

## Troubleshooting
| Problem | Fix |
|---|---|
| Health page does not open on the Quest | same network? hotspot on? firewall rule added? |
| "Cleartext HTTP not permitted" | allow HTTP downloads in Player Settings |
| No boxes, server log shows requests | instrument too small/far - move closer; check `/health` model is loaded |
| HTTP 503 | server has no model weights (models/deploy missing) |
| Slow (> 1 s) | lower Max Side to 960 in the client; training on the same PC slows the server |
