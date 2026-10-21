import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image
from src.cv.structure_rules import detect_structure, connect_arrows
from src.hybrid.document import build_document, layout_candidates
from src.evaluation.annotation_files import create, merge


def page(): return np.full((600,800,3),255,np.uint8)
def table(image,merged=False):
    for y in (100,200,300): cv2.line(image,(100,y),(500,y),(0,0,0),2)
    for x in (100,500): cv2.line(image,(x,100),(x,300),(0,0,0),2)
    cv2.line(image,(300,200 if merged else 100),(300,300),(0,0,0),2)

class StructureRulesTests(unittest.TestCase):
    def test_grid_and_text_assignment_and_merged_cell(self):
        for merged in (False,True):
            im=page(); table(im,merged)
            result=detect_structure(im,[{'id':'t','text':'한국어','bbox':[120,120,210,150]},
                                       {'id':'cross','text':'cross','bbox':[250,220,350,260]}])
            self.assertEqual(len(result['tables']),1)
            t=result['tables'][0]; self.assertEqual((t['rows'],t['cols']),(2,2))
            self.assertEqual(len(t['cells']),3 if merged else 4)
            self.assertEqual(sum('t' in c['text_ids'] for c in t['cells']),1)
            self.assertIn('cross',result['unassigned_text_ids'])
            self.assertEqual(max(c['colspan'] for c in t['cells']),2 if merged else 1)

    def test_plain_lines_box_and_blank_are_not_structures(self):
        for kind in ('blank','box','lines','shaft'):
            im=page()
            if kind=='box':cv2.rectangle(im,(100,100),(500,300),(0,0,0),2)
            if kind=='lines':
                for y in (100,200,300):cv2.line(im,(50,y),(750,y),(0,0,0),2)
            if kind=='shaft':cv2.line(im,(100,450),(600,450),(0,0,0),2)
            result=detect_structure(im)
            self.assertEqual(result['tables'],[],kind); self.assertEqual(result['arrows'],[],kind)

    def test_arrow_direction_and_bidirectional(self):
        for start,end in [((150,450),(550,450)),((550,450),(150,450)),((650,100),(650,500))]:
            im=page();cv2.arrowedLine(im,start,end,(0,0,0),2,tipLength=.07)
            arrows=detect_structure(im)['arrows'];self.assertEqual(len(arrows),1)
            a=arrows[0]
            self.assertLess(np.linalg.norm(np.array(a['end'])-end),8)
            self.assertIsNone(a['target_id'])
        im=page();cv2.arrowedLine(im,(150,450),(550,450),(0,0,0),2,tipLength=.07)
        cv2.arrowedLine(im,(550,450),(150,450),(0,0,0),2,tipLength=.07)
        self.assertEqual(detect_structure(im)['arrows'][0]['direction'],'bidirectional')

    def test_connection_direction_and_tie_abstention(self):
        arrow={'start':[200,300],'end':[500,300]}
        targets=[{'id':'s','bbox':[100,270,190,330]},{'id':'t','bbox':[510,270,600,330]}]
        connect_arrows([arrow],targets,(800,600))
        self.assertEqual((arrow['source_id'],arrow['target_id']),('s','t'))
        connect_arrows([arrow],targets+[{'id':'ambiguous','bbox':[510,270,600,330]}],(800,600))
        self.assertIsNone(arrow['target_id'])

    def test_rescale_and_document_uses_corrected_cell_text(self):
        im=page();table(im)
        im=cv2.resize(im,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
        lines=[{'id':'ocr-1','text':'원문','bbox':[360,360,600,450]}]
        layout=layout_candidates(im,lines)
        tables=[t for t in layout if t['type']=='Table'];self.assertEqual(len(tables),1)
        self.assertLess(abs(tables[0]['bbox'][0]-300),6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.png';Image.fromarray(im).save(path)
            doc=build_document(path,{'lines':lines,'layout':layout},{'vector_edits':[{'id':'ocr-1','text':'교정'}]},'test')
            self.assertFalse(any(e['id']=='ocr-1' for e in doc['elements']))
            cells=[e for e in doc['elements'] if e.get('semantic_type')=='table-cell']
            self.assertEqual(len(cells),4)
            self.assertIn('교정',[e['text'] for e in cells])
            self.assertTrue(all('rowspan' in e for e in cells))

    def test_text_endpoint_links_do_not_violate_flow_schema(self):
        from src.web.backend.app import _validate_document
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.png';Image.new('RGB',(800,600),'white').save(path)
            lines=[{'id':'s','text':'source','bbox':[100,270,190,330]},
                   {'id':'t','text':'target','bbox':[510,270,600,330]}]
            arrow={'id':'a','type':'Arrow','source':'structure-rules','bbox':[200,299,500,301],
                   'start':[200,300],'end':[500,300],'direction':'forward'}
            doc=build_document(path,{'lines':lines,'layout':[arrow]},{'vector_edits':[]},'test')
            _validate_document(doc)
            a=next(e for e in doc['elements'] if e['id']=='a')
            self.assertEqual((a['source_id'],a['target_id']),('s','t'))
            self.assertIsNone(doc['relations'][0]['source'])

class AnnotationFilesTests(unittest.TestCase):
    def test_preserve_manual_edits_and_validate_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/'manifest.json'; refs=root/'refs.json'; labels=root/'labels'
            manifest.write_text(json.dumps({'samples':[{'id':'hf-0000'},{'id':'hf-0001'}]}))
            refs.write_text(json.dumps({'samples':[{'id':'hf-0000','text':'검수','status':'verified','annotator':'A','notes':''}]}))
            self.assertEqual(create(manifest,refs,labels),2)
            p=labels/'hf-0001.json'; label=json.loads(p.read_text());label.update(text='manual',status='draft');p.write_text(json.dumps(label))
            self.assertEqual(create(manifest,refs,labels),0)
            self.assertEqual(merge(labels,manifest,root/'merged.json'),2)
            self.assertEqual(json.loads((root/'merged.json').read_text())['samples'][1]['text'],'manual')
            label.update(id='wrong');p.write_text(json.dumps(label))
            with self.assertRaises(ValueError):merge(labels,manifest,root/'merged.json')

if __name__=='__main__':unittest.main()
