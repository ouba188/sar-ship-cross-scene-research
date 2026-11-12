from pathlib import Path
import win32com.client

root=Path(__file__).resolve().parent/'expanded'
app=win32com.client.Dispatch('PowerPoint.Application')
deck=app.Presentations.Open(str(root/'L1-L3_同船12457_完整干预图_可编辑.pptx'),ReadOnly=True,WithWindow=False)
try:
    out=root/'preview';out.mkdir(exist_ok=True)
    for i in range(1,deck.Slides.Count+1):
        deck.Slides(i).Export(str(out/f'slide_{i:02d}.png'),'PNG',1800,1000)
    deck.SaveAs(str(root/'L1-L3_同船12457_完整干预图_预览.pdf'),32)
    print(f'Rendered {deck.Slides.Count} slides')
finally:
    deck.Close()
