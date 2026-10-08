// Unity client for the oscilloscope recognition server (Leonardo AR contract).
//
// Sends camera frames to POST /v1/recognitions, keeps only the newest answer (stale responses are
// discarded, as the Leonardo notes require), and raises an event per frame with the instruments found.
// Each recognised model (rs_rtb2004, tek_tds2014, tek_tds1002) can be mapped to the AR module to show.
//
// Setup: add to any GameObject, set Server Url to the PC on the LAN (e.g. http://192.168.1.20:8011),
// and assign a camera source: either a WebCamTexture (Quest passthrough camera via Meta's Passthrough
// Camera API exposes it as a WebCamTexture) or call SubmitFrame(texture) from your own capture code.
// Requires: Unity 2021.3+, Newtonsoft Json (com.unity.nuget.newtonsoft-json).
using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace LeonardoAR
{
    [Serializable]
    public class BBox { public float x, y, width, height; }   // normalised 0..1, origin top-left

    [Serializable]
    public class Detection
    {
        public string class_id;
        public string label;          // rs_rtb2004 | tek_tds2014 | tek_tds1002 (targets), COCO names otherwise
        public string display_name;
        public float confidence;
        public BBox bbox;
        public bool target;           // true = one of the PoC oscilloscopes
    }

    [Serializable]
    public class RecognitionResponse
    {
        public string request_id;
        public string session_id;
        public long? frame_timestamp_ms;
        public string mode;           // "real" | "simulated"
        public bool simulated;
        public float inference_ms;
        public List<Detection> detections = new List<Detection>();
    }

    [Serializable]
    public class ModuleBinding
    {
        public string label = "rs_rtb2004";
        public GameObject arModule;   // AR content to enable when this instrument is recognised
    }

    [Serializable]
    public class RecognitionEvent : UnityEvent<RecognitionResponse> { }

    public class OscilloscopeRecognitionClient : MonoBehaviour
    {
        [Header("Server (LAN only - never expose it to the Internet)")]
        public string serverUrl = "http://192.168.1.20:8011";
        [Tooltip("Frames per second sent to the server; the CPU server needs ~0.3-1 s per frame.")]
        [Range(0.2f, 5f)] public float sendRate = 1f;
        [Range(30, 95)] public int jpegQuality = 75;
        [Tooltip("Longest image side sent; smaller is faster on Wi-Fi. The server works well from 960 px.")]
        public int maxSide = 1280;
        [Tooltip("Ignore results older than this (ms); the headset may have moved since.")]
        public int maxResultAgeMs = 1500;
        [Tooltip("Accept simulated results (server in RECOGNITION_MODE=simulated). Keep off in real use.")]
        public bool acceptSimulated = false;
        [Tooltip("Learning from real use: let the lab server keep frames where the model is unsure, for labelling. " +
                 "Only has an effect if the server runs with CAPTURE_MODE=on. Off by default (frames are not stored).")]
        public bool contributeUnsureFrames = false;

        [Header("Camera source (optional)")]
        public WebCamTexture webcam;

        [Header("AR modules per instrument")]
        public List<ModuleBinding> modules = new List<ModuleBinding>();

        [Header("Events")]
        public RecognitionEvent onRecognition = new RecognitionEvent();

        public RecognitionResponse Latest { get; private set; }
        public bool ServerReachable { get; private set; }

        string sessionId;
        long newestSentTs;
        long newestAppliedTs;
        bool inFlight;
        Texture2D scratch;

        void Start()
        {
            sessionId = SystemInfo.deviceUniqueIdentifier.Substring(0, 8) + "-" + DateTime.UtcNow.ToString("HHmmss");
            StartCoroutine(CheckHealth());
            if (webcam != null) StartCoroutine(Loop());
        }

        IEnumerator CheckHealth()
        {
            using (var req = UnityWebRequest.Get(serverUrl.TrimEnd('/') + "/health"))
            {
                req.timeout = 5;
                yield return req.SendWebRequest();
                ServerReachable = req.result == UnityWebRequest.Result.Success;
                Debug.Log(ServerReachable ? $"[Recognition] server ok: {req.downloadHandler.text}"
                                          : $"[Recognition] server not reachable: {req.error}");
            }
        }

        /// Attach a camera at runtime (e.g. from CameraStarter) and start sending frames.
        public void UseCamera(WebCamTexture cam)
        {
            bool running = webcam != null;
            webcam = cam;
            if (!running) StartCoroutine(Loop());
        }

        IEnumerator Loop()
        {
            var wait = new WaitForSeconds(1f / sendRate);
            while (enabled)
            {
                // (didUpdateThisFrame is unreliable inside a coroutine tick, so only check isPlaying)
                if (!inFlight && webcam.isPlaying && webcam.width > 16) SubmitFrame(webcam);
                yield return wait;
            }
        }

        /// Send one frame. Works with WebCamTexture, RenderTexture or Texture2D.
        public void SubmitFrame(Texture source)
        {
            if (inFlight) return;               // one request at a time: the CPU server is the bottleneck
            byte[] jpg = Encode(source);
            long ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            newestSentTs = ts;
            StartCoroutine(Send(jpg, ts));
        }

        byte[] Encode(Texture source)
        {
            float s = Mathf.Min(1f, (float)maxSide / Mathf.Max(source.width, source.height));
            int w = Mathf.RoundToInt(source.width * s), h = Mathf.RoundToInt(source.height * s);
            var rt = RenderTexture.GetTemporary(w, h, 0);
            Graphics.Blit(source, rt);
            if (scratch == null || scratch.width != w || scratch.height != h)
                scratch = new Texture2D(w, h, TextureFormat.RGB24, false);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            scratch.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            scratch.Apply();
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
            return scratch.EncodeToJPG(jpegQuality);
        }

        IEnumerator Send(byte[] jpg, long ts)
        {
            inFlight = true;
            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("image", jpg, "frame.jpg", "image/jpeg"),
                new MultipartFormDataSection("session_id", sessionId),
                new MultipartFormDataSection("frame_timestamp_ms", ts.ToString()),
            };
            if (contributeUnsureFrames) form.Add(new MultipartFormDataSection("capture", "true"));
            using (var req = UnityWebRequest.Post(serverUrl.TrimEnd('/') + "/v1/recognitions", form))
            {
                req.timeout = 10;
                yield return req.SendWebRequest();
                inFlight = false;
                if (req.result != UnityWebRequest.Result.Success)
                {
                    // 503 = server up but model weights missing; no fallback, as in the Leonardo notes
                    Debug.LogWarning($"[Recognition] HTTP {req.responseCode}: {req.error}");
                    yield break;
                }
                var res = JsonConvert.DeserializeObject<RecognitionResponse>(req.downloadHandler.text);
                Apply(res, ts);
            }
        }

        void Apply(RecognitionResponse res, long ts)
        {
            if (res == null) return;
            if (res.simulated && !acceptSimulated) return;
            if (res.session_id != sessionId) return;                       // not ours
            if (ts < newestAppliedTs) return;                              // an older frame answered late
            if (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - ts > maxResultAgeMs) return;  // stale
            newestAppliedTs = ts;
            Latest = res;

            var seen = new HashSet<string>();
            foreach (var d in res.detections)
                if (d.target) seen.Add(d.label);
            foreach (var m in modules)
                if (m.arModule != null) m.arModule.SetActive(seen.Contains(m.label));
            onRecognition.Invoke(res);
        }

        /// Box of the best detection of a given instrument, in viewport coordinates (origin bottom-left),
        /// handy for placing a UI anchor; null if that instrument is not in the latest result.
        public Rect? ViewportRectOf(string label)
        {
            if (Latest == null) return null;
            Detection best = null;
            foreach (var d in Latest.detections)
                if (d.target && d.label == label && (best == null || d.confidence > best.confidence)) best = d;
            if (best == null) return null;
            return new Rect(best.bbox.x, 1f - best.bbox.y - best.bbox.height, best.bbox.width, best.bbox.height);
        }
    }
}
