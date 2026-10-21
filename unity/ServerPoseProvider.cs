// Pose provider for the Leonardo AR PoC: Recognition -> Pose (server) -> Unity reference objects.
//
// Sends camera frames to the server's POST /v1/pose (recognition + 6D pose, see docs/POSE_STAGE.md) and places the
// reference object (prefab with the local AR points) of the recognised model at the returned pose.
// It raises the same kind of event as a simulated provider (label, world position, world rotation), so it can
// replace the simulated pose provider without touching the AR content.
//
// Frames: the passthrough camera as a WebCamTexture (UseCamera) or your own capture via SubmitFrame(texture, camPose).
// Intrinsics: fx, fy, cx, cy of the camera image at its FULL resolution (Passthrough Camera API helpers); they are
// rescaled here when the frame is downscaled before sending.
// Reference objects must use the frame of the server's mesh: import models/meshes/<label>/<label>.obj into Unity
// (Unity mirrors x on import; the server's "unity" pose already accounts for that) or build the prefab so that its
// local origin is the centre of the instrument box, +y up and the front panel facing local -z after import.
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace LeonardoAR
{
    [Serializable]
    public class UnityPose { public float[] position; public float[] rotation; }    // metres; quaternion x, y, z, w

    [Serializable]
    public class PoseResult
    {
        public string state;          // registered | tracking | lost | no_target | needs_depth
        public string label;          // rs_rtb2004 / tek_tds2014 / tek_tds1002
        public string display_name;
        public string backend;        // foundationpose | geometric
        public bool approximate;      // true for the CPU fallback (orientation assumed "front facing the camera")
        public float[] ob_in_cam;     // 4x4 row-major, OpenCV camera frame (for debugging / other clients)
        public UnityPose unity;       // the same pose in Unity camera space, for a prefab imported from our OBJ
        public float box_agreement;   // overlap of the posed 3D box with the detector box (0..1)
        public string reason;
    }

    [Serializable]
    public class PoseResponse
    {
        public string request_id;
        public string session_id;
        public long frame_timestamp_ms;
        public PoseResult pose;
        public float total_ms;
    }

    [Serializable]
    public class ReferenceBinding
    {
        public string label = "tek_tds1002";
        public Transform referenceObject;     // prefab instance with the local points for the AR content
    }

    [Serializable]
    public class PoseEvent : UnityEvent<string, Vector3, Quaternion> { }   // label, world position, world rotation

    public class ServerPoseProvider : MonoBehaviour
    {
        [Header("Server (LAN only)")]
        public string serverUrl = "http://192.168.1.20:8011";
        [Range(0.2f, 10f)] public float sendRate = 2f;
        [Range(30, 95)] public int jpegQuality = 80;
        public int maxSide = 1280;
        [Tooltip("Ask for this model only (empty = the best recognised one).")]
        public string onlyLabel = "";

        [Header("Camera")]
        public WebCamTexture webcam;
        [Tooltip("Transform of the physical camera at capture time (passthrough camera pose); Main Camera if empty.")]
        public Transform cameraTransform;
        [Tooltip("Intrinsics at the camera's full resolution. 0 = let the server guess (only for tests).")]
        public float fx, fy, cx, cy;
        public int intrinsicsWidth = 1280;

        [Header("Reference objects")]
        public List<ReferenceBinding> references = new List<ReferenceBinding>();
        [Tooltip("Smooth the pose over frames (0 = raw, 0.9 = very smooth).")]
        [Range(0f, 0.95f)] public float smoothing = 0.5f;

        public PoseEvent onPose = new PoseEvent();
        public PoseResponse Latest { get; private set; }

        readonly string session = Guid.NewGuid().ToString("N");
        Texture2D scratch;
        bool busy;

        void Start() { StartCoroutine(Loop()); }

        public void UseCamera(WebCamTexture cam) { webcam = cam; }

        public void ResetTracking() { StartCoroutine(Send(null, Pose.identity, true)); }

        IEnumerator Loop()
        {
            while (true)
            {
                yield return new WaitForSeconds(1f / sendRate);
                if (webcam != null && webcam.didUpdateThisFrame && !busy)
                {
                    var cam = cameraTransform != null ? cameraTransform : Camera.main.transform;
                    SubmitFrame(webcam, new Pose(cam.position, cam.rotation));
                }
            }
        }

        /// Send one frame together with the camera pose at the moment it was captured.
        public void SubmitFrame(Texture source, Pose cameraPoseAtCapture)
        {
            if (busy) return;
            StartCoroutine(Send(source, cameraPoseAtCapture, false));
        }

        IEnumerator Send(Texture source, Pose camPose, bool reset)
        {
            busy = true;
            var form = new WWWForm();
            form.AddField("session_id", session);
            form.AddField("frame_timestamp_ms", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString());
            if (!string.IsNullOrEmpty(onlyLabel)) form.AddField("label", onlyLabel);
            if (reset) form.AddField("reset", "true");
            float scale = 1f;
            if (source != null)
            {
                byte[] jpg = Encode(source, out int w);
                scale = (float)w / Mathf.Max(1, intrinsicsWidth);
                form.AddBinaryData("image", jpg, "frame.jpg", "image/jpeg");
            }
            if (fx > 0)
            {
                var inv = System.Globalization.CultureInfo.InvariantCulture;
                form.AddField("fx", (fx * scale).ToString(inv));
                form.AddField("fy", ((fy > 0 ? fy : fx) * scale).ToString(inv));
                form.AddField("cx", (cx * scale).ToString(inv));
                form.AddField("cy", (cy * scale).ToString(inv));
            }
            if (source == null) { busy = false; yield break; }   // reset only takes effect with the next frame
            using (var req = UnityWebRequest.Post(serverUrl.TrimEnd('/') + "/v1/pose", form))
            {
                req.timeout = 10;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    Latest = JsonUtility.FromJson<PoseResponse>(req.downloadHandler.text);
                    Apply(Latest, camPose);
                }
                else Debug.LogWarning("pose request failed: " + req.error);
            }
            busy = false;
        }

        void Apply(PoseResponse r, Pose camPose)
        {
            var p = r?.pose;
            if (p == null || p.unity == null || p.unity.position == null || p.unity.position.Length != 3) return;
            if (p.state != "registered" && p.state != "tracking") return;
            var localPos = new Vector3(p.unity.position[0], p.unity.position[1], p.unity.position[2]);
            var localRot = new Quaternion(p.unity.rotation[0], p.unity.rotation[1], p.unity.rotation[2], p.unity.rotation[3]);
            // camera space -> world, with the camera pose of the frame (the head may have moved since)
            var worldPos = camPose.position + camPose.rotation * localPos;
            var worldRot = camPose.rotation * localRot;
            foreach (var b in references)
            {
                if (b.referenceObject == null || b.label != p.label) continue;
                var t = b.referenceObject;
                bool first = !t.gameObject.activeSelf;
                t.gameObject.SetActive(true);
                float k = first ? 0f : smoothing;
                t.SetPositionAndRotation(Vector3.Lerp(worldPos, t.position, k), Quaternion.Slerp(worldRot, t.rotation, k));
            }
            onPose.Invoke(p.label, worldPos, worldRot);
        }

        byte[] Encode(Texture source, out int width)
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
            width = w;
            return scratch.EncodeToJPG(jpegQuality);
        }
    }
}
