// Debug overlay: draws the recognised instruments as coloured boxes with names on screen.
// Add next to OscilloscopeRecognitionClient (same or any GameObject) and assign the client.
// Boxes are drawn over the full screen, so they line up when the camera image fills the view
// (e.g. the editor preview with CameraStarter's RawImage, or a passthrough feed shown full screen).
using UnityEngine;

namespace LeonardoAR
{
    public class RecognitionOverlay : MonoBehaviour
    {
        public OscilloscopeRecognitionClient client;
        public bool showOtherObjects = false;
        [Range(10, 40)] public int fontSize = 18;

        static readonly Color Rtb = new Color(0.09f, 0.76f, 0.5f);
        static readonly Color Tek2014 = new Color(1f, 0.54f, 0.12f);
        static readonly Color Tek1002 = new Color(0.82f, 0.29f, 0.82f);
        GUIStyle style;
        Texture2D white;

        void OnGUI()
        {
            if (client == null || client.Latest == null) return;
            if (style == null)
            {
                style = new GUIStyle(GUI.skin.label) { fontSize = fontSize, fontStyle = FontStyle.Bold };
                white = Texture2D.whiteTexture;
            }
            foreach (var d in client.Latest.detections)
            {
                if (!d.target && !showOtherObjects) continue;
                var c = !d.target ? Color.gray : d.label == "rs_rtb2004" ? Rtb : d.label == "tek_tds2014" ? Tek2014 : Tek1002;
                var r = new Rect(d.bbox.x * Screen.width, d.bbox.y * Screen.height,
                                 d.bbox.width * Screen.width, d.bbox.height * Screen.height);
                DrawFrame(r, c, d.target ? 4 : 2);
                var text = $"{d.display_name ?? d.label}  {Mathf.RoundToInt(d.confidence * 100)}%";
                var size = style.CalcSize(new GUIContent(text));
                var tag = new Rect(r.x, Mathf.Max(0, r.y - size.y), size.x + 8, size.y);
                GUI.color = c; GUI.DrawTexture(tag, white); GUI.color = Color.black;
                GUI.Label(new Rect(tag.x + 4, tag.y, tag.width, tag.height), text, style);
                GUI.color = Color.white;
            }
        }

        void DrawFrame(Rect r, Color c, float t)
        {
            GUI.color = c;
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, t), white);
            GUI.DrawTexture(new Rect(r.x, r.yMax - t, r.width, t), white);
            GUI.DrawTexture(new Rect(r.x, r.y, t, r.height), white);
            GUI.DrawTexture(new Rect(r.xMax - t, r.y, t, r.height), white);
            GUI.color = Color.white;
        }
    }
}
