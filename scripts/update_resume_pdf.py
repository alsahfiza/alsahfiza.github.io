"""Regenerate assets/Resume.pdf from _data/resume.yml. Requires PyYAML and reportlab."""
from pathlib import Path
import yaml
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,KeepTogether,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
r=Path(__file__).resolve().parents[1];d=yaml.safe_load((r/'_data/resume.yml').read_text());b=d['basics'];styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Detail',fontName='DejaVu',fontSize=9,leading=12,spaceAfter=5))
styles['Title'].fontName='DejaVu-Bold';styles['Heading1'].fontName='DejaVu-Bold';styles['Heading2'].fontName='DejaVu-Bold'
styles['Heading1'].textColor=colors.HexColor('#17365a');styles['Heading1'].fontSize=14;styles['Heading1'].spaceBefore=14
styles['Heading2'].fontSize=11;styles['Heading2'].leading=14
story=[]
def p(s,style='Detail'):return Paragraph(escape(str(s)).replace('\n','<br/>'),styles[style])
story.extend([p(b['name'],'Title'),p(b['label']),p(b['email']+' | '+b['phone']+' | '+b['website'])])
for x in b['profiles']:story.append(p(x['network']+': '+x['url']))
for key,title in [('work','Experience'),('education','Education'),('certificate','Certificates'),('publications','Publication')]:
 if key=='certificate':story.append(PageBreak())
 story.append(p(title,'Heading1'))
 for x in d[key]:
  name=x.get('position') or x.get('area') or x.get('name');org=x.get('company') or x.get('institution') or x.get('publisher')
  date=x.get('dateLabel') or x.get('date') or x.get('releaseDate') or (x.get('startDate','')+' - '+(x.get('endDate') or 'Present'))
  if key=='publications':date='Journal: '+str(x['publicationYear'])+'; Research Square preprint: '+x['preprintDate']
  story.append(KeepTogether([p(name,'Heading2'),p(org+' | '+date)]))
  if x.get('studyType'):story.append(p(x['studyType']))
  if x.get('summary'):story.append(p(x['summary']))
  for h in x.get('highlights',[]):story.append(p('- '+h))
  if x.get('website'):story.append(p(x['website']))
SimpleDocTemplate(str(r/'assets/Resume.pdf'),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=38,title='Zakaria Alsahfi - Resume',author=b['name']).build(story)
print('Resume regenerated from YAML')
