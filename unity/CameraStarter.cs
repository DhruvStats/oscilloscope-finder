// Starts a camera as a WebCamTexture and hands it to the recognition client.
// In the Unity editor this is the PC webcam (quick test without a headset); on Quest 3 / 3S with Meta's
// Passthrough Camera API enabled (camera permission granted) the headset camera also appears as a WebCamTexture.
// Optional: assign a full-screen RawImage to see the camera image under the RecognitionOverlay boxes.
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace LeonardoAR
{
    public class CameraStarter : MonoBehaviour
    {
        public OscilloscopeRecognitionClient client;
        [Tooltip("Part of the device name to pick (empty = first camera).")]
        public string deviceNameContains = "";
        public int requestedWidth = 1280, requestedHeight = 960;
        public RawImage preview;

        IEnumerator Start()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
            {
                UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera);
                // Quest passthrough camera also needs: horizonos.permission.HEADSET_CAMERA (see unity/SETUP.md)
                UnityEngine.Android.Permission.RequestUserPermission("horizonos.permission.HEADSET_CAMERA");
                yield return new WaitForSeconds(1f);
            }
#endif
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
            var devices = WebCamTexture.devices;
            if (devices.Length == 0) { Debug.LogError("[CameraStarter] no camera found"); yield break; }
            var dev = devices[0].name;
            foreach (var d in devices)
                if (!string.IsNullOrEmpty(deviceNameContains) && d.name.Contains(deviceNameContains)) { dev = d.name; break; }
            var cam = new WebCamTexture(dev, requestedWidth, requestedHeight, 30);
            cam.Play();
            while (cam.width <= 16) yield return null;
            Debug.Log($"[CameraStarter] using '{dev}' {cam.width}x{cam.height}");
            if (preview != null) preview.texture = cam;
            client.UseCamera(cam);
        }
    }
}
