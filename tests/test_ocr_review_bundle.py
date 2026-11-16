import copy,json,tempfile,unittest
from pathlib import Path
from src.ocr_data.review_bundle import Store,validate,write_new,sha256
from src.evaluation.ocr_batch_contract import verify

RUN=Path('data/experiments/ocr_review_batch_20261111')
class ReviewBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        base=json.load(open(RUN/'review/draft.json'));base['pages']=base['pages'][:1];base['pages'][0]['regions']=base['pages'][0]['regions'][:2]
        write_new(self.root/'draft.json',base);write_new(self.root/'input-lock.json',{'hashes':{}})
        self.store=Store(self.root,test_mode=True);self.page=base['pages'][0]['image_id'];self.ids=[r['annotation_id'] for r in base['pages'][0]['regions']]
    def tearDown(self):self.temp.cleanup()
    def act(self,action,ids=[],**kw):return self.store.action(dict(action=action,annotation_ids=ids,image_id=self.page,revision=self.store.state['revision'],reason='TEST fixture evidence',**kw))
    def values(self):return dict(bbox=[0,0,20,20],text='한글 English',type='text',include=True,ignore_reason=None,order=1,uncertainty=None,active=True)
    def test_risk_sources_exactly_cover_73_and_60_unique(self):
        risk=json.load(open(RUN/'review/risk-items.json'));self.assertEqual(risk['unique_items'],60);self.assertFalse(risk['unresolved_links'])
        self.assertEqual(sum(len(r['sources']) for r in risk['items']),73);self.assertEqual(len({(r['image_id'],r['annotation_id']) for r in risk['items']}),60)
        self.assertEqual(sum(any(s['source']=='FP13' for s in r['sources']) for r in risk['items']),13)
    def test_save_restart_export_import_exact(self):
        self.act('edit',[self.ids[0]],values=self.values());s=copy.deepcopy(self.store.state);self.store.save()
        restarted=Store(self.root,True);self.assertEqual(restarted.state,s);self.assertEqual(restarted.restore(json.loads(json.dumps(s))),s)
        self.assertEqual(Store(self.root,True).state,s)
    def test_add_split_merge_hold_roundtrip_lineage(self):
        self.act('add',regions=[self.values()]);self.act('split',[self.ids[0]],regions=[self.values(),dict(self.values(),order=1.5,uncertainty='unclear')])
        children=[r['annotation_id'] for r in self.store.state['pages'][0]['regions'][-2:]]
        self.assertEqual(self.store.state['pages'][0]['regions'][-1]['review_status'],'hold')
        self.act('merge',children,regions=[self.values()]);s=copy.deepcopy(self.store.state);self.assertEqual(Store(self.root,True).state,s)
        self.assertEqual(s['pages'][0]['regions'][-1]['lineage'][0]['parents'],children);self.assertFalse(s['pages'][0]['regions'][0]['active'])
    def test_no_human_attribution_without_action_and_attestation(self):
        s=copy.deepcopy(self.store.state);r=s['pages'][0]['regions'][0];r.update(human_verified=True,reviewer='fake',reviewed_at='now')
        with self.assertRaises(ValueError):validate(s,self.store.baseline)
        with self.assertRaises(ValueError):self.act('verify_region',[self.ids[0]],reviewer='TEST',human_attested=False)
        self.act('edit',[self.ids[0]],values=self.values());self.act('verify_region',[self.ids[0]],reviewer='TEST',human_attested=True)
        self.assertFalse(self.store.state['pages'][0]['regions'][0]['human_verified'])
    def test_partial_region_review_cannot_complete_page(self):
        self.act('edit',[self.ids[0]],values=self.values());self.act('verify_region',[self.ids[0]],reviewer='TEST',human_attested=True)
        self.assertEqual(self.store.state['pages'][0]['review_status'],'in_progress')
        with self.assertRaises(ValueError):self.act('complete_page',reviewer='TEST',human_attested=True)
        self.act('edit',[self.ids[1]],values=dict(self.values(),order=2));self.act('verify_region',[self.ids[1]],reviewer='TEST',human_attested=True)
        with self.assertRaises(ValueError):self.act('complete_page',reviewer='TEST',human_attested=True)
        self.act('check_page',checks={k:True for k in self.store.state['pages'][0]['completeness']});self.act('complete_page',reviewer='TEST',human_attested=True)
        self.assertEqual(self.store.state['pages'][0]['review_status'],'complete');self.assertFalse(self.store.state['pages'][0]['human_verified'])
    def test_source_and_cache_drift_rejected(self):
        self.act('edit',[self.ids[0]],values=self.values());s=copy.deepcopy(self.store.state);s['pages'][0]['regions'][0]['ai_proposal']['text']='changed'
        with self.assertRaises(ValueError):self.store.restore(s)
        with self.assertRaises(ValueError):self.store.action(dict(action='edit',revision=0,image_id=self.page,annotation_ids=[self.ids[0]],values=self.values()))
        real=copy.deepcopy(self.store.state);real['test_mode']=False
        with self.assertRaises(ValueError):self.store.restore(real)
    def test_invalid_bbox_or_uncertainty_cannot_verify(self):
        with self.assertRaises(ValueError):self.act('edit',[self.ids[0]],values=dict(self.values(),bbox=[0,0,-1,20]))
        self.act('edit',[self.ids[0]],values=dict(self.values(),uncertainty='unclear'))
        with self.assertRaises(ValueError):self.act('verify_region',[self.ids[0]],reviewer='TEST',human_attested=True)
    def test_batch_plan_fixed_members_order_and_drift(self):
        plan=json.load(open(RUN/'batch/plan.json'));self.assertEqual(len(plan['changed13']),13);self.assertEqual(len(plan['targets']),15)
        for target in plan['targets']:
            rows={r['condition']:r for r in plan['matrix'] if r['target']==target}
            self.assertEqual(rows['original297']['members'],rows['original297_repeat']['members']);self.assertEqual(rows['original297']['members'][::-1],rows['reordered']['members'])
            self.assertEqual(rows['single']['members'],[target]);self.assertEqual(rows['single_repeat']['members'],[target])
        write_new(self.root/'plan.json',plan);write_new(self.root/'batch-adapter',{'weight':'original'})
        lock=dict(config=json.load(open('configs/ocr_v2_vl_lora.json')),hashes={str(self.root/'batch-adapter'):sha256(self.root/'batch-adapter')},plan_sha256=sha256(self.root/'plan.json'))
        write_new(self.root/'input-lock-batch.json',lock)
        # verify uses fixed path; isolate from review store.
        b=self.root/'batch';b.mkdir();write_new(b/'plan.json',plan);write_new(b/'input-lock.json',lock);verify(b)
        (self.root/'batch-adapter').write_text('drift')
        with self.assertRaises(ValueError):verify(b)
    def test_import_rejects_empty_completeness_and_integer_include(self):
        s=copy.deepcopy(self.store.state);s['pages'][0]['completeness']={}
        with self.assertRaises(ValueError):self.store.restore(s)
        with self.assertRaises(ValueError):self.act('edit',[self.ids[0]],values=dict(self.values(),include=1))
    def test_singleton_contract_id_sort_and_crop_drift(self):
        from src.cv.ocr_research_infer import prepare_rows
        crop=self.root/'crop.png';crop.write_bytes(b'fixed bytes')
        row=dict(page_id='p',region_id='b',image=str(crop),crop_sha256=sha256(crop),text='')
        path=self.root/'rows.json';write_new(path,[row,dict(row,region_id='a')]);rows=prepare_rows(path,self.root)
        self.assertEqual([r['region_id'] for r in rows],['a','b'])
        path2=self.root/'duplicate.json';write_new(path2,[row,row])
        with self.assertRaises(ValueError):prepare_rows(path2,self.root)
        crop.write_bytes(b'drift')
        with self.assertRaises(ValueError):prepare_rows(path,self.root)
    def test_full_bundle_build_stays_unreviewed_and_exclusive(self):
        from src.ocr_data.review_bundle import build
        output=self.root/'new-bundle';build(output);draft=json.load(open(output/'draft.json'))
        self.assertEqual(len(draft['pages']),17);self.assertEqual(sum(len(p['regions']) for p in draft['pages']),479)
        self.assertTrue((output/'schema-v2.json').is_file());self.assertTrue(all(not r['human_verified'] and r['reviewer'] is None and r['reviewed_at'] is None and not r['text'] for p in draft['pages'] for r in p['regions']))
        with self.assertRaises(FileExistsError):build(output)
