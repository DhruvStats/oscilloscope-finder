"""Write the Label Studio labelling interface from config/instruments.yaml.

Run after adding an instrument to the registry:
    .venv/Scripts/python tools/registry_sync.py
Then paste labelstudio/labeling_config.xml into the project (Settings > Labeling Interface), or create a
new project with labelstudio/setup_project.py. Hotkeys 1, 2, 3 ... follow the registry order.
"""
import os
import sys
from xml.sax.saxutils import quoteattr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import registry  # noqa: E402


def main():
    items = registry.load()
    labels = "\n".join(
        f'    <Label value={quoteattr(it["label"])} background={quoteattr(it["colour"])}'
        + (f' hotkey="{i + 1}"' if i < 9 else "") + "/>" for i, it in enumerate(items))
    xml = f"""<View>
  <Header value="Box every one of the target instruments that is visible (even partly). Leave everything else unboxed."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3">
{labels}
  </RectangleLabels>
  <Header value="Photo status" size="5"/>
  <Choices name="status" toName="image" choice="single" showInline="true">
    <Choice value="done" hotkey="d"/>
    <Choice value="no target oscilloscope"/>
    <Choice value="unsure which model"/>
    <Choice value="exclude (too blurry / too small)"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Notes (optional)" rows="1" maxSubmissions="1"/>
</View>
"""
    out = os.path.join(ROOT, "labelstudio", "labeling_config.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"{len(items)} instruments -> {os.path.relpath(out, ROOT)}")
    for i, it in enumerate(items):
        print(f"  key {i + 1}: {it['label']:<14} {it['display_name']}")


if __name__ == "__main__":
    main()
