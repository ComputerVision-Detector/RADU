"""AI visual review notes after all17 originals and all201 FP crops inspected.
Texts below transcribed from image sheets, not recognition output. No human validation.
Indices refer to frozen-fp201.json. Detection-assisted new geometry is provisional.
"""
# Primary classes for each fixed FP; default only within explicitly inspected ranges.
CATEGORIES={
 'split_merge_iou':[0,33,48,80,81,84,85,86],
 'excluded_region':[2,4,5,7,8,13,15,21,23,25,27,28,29,31,34,40,41,42,43,44,54,62,63,65,66,68,83,88,97,98,99,100,101,103,104,105,106,107,109,110,111,117,118,119,121,122,123,124,125,126,127,130,146,152,163,165,168,170,171,172,173,174,175,177,178,179,180,181,183,184,185,187,188,189,191,192,194,195,196,197,199,200],
 'duplicate':[36,47,176],
 'actual_nontext':[164],
 'uncertain':[89,90,92,93,94,95,114,116,128,132,139,141,153],
}
# Remaining inspected indices are genuine normal text not represented in legacy199 scope.
# Subclassification will distinguish original V2 omission vs CTC scope exclusion geometrically.
NEIGHBOR_TAGS=[55,56,73,74,93,114,132,138,141,176]
TRANSCRIPTS={
1:'Linear 4096 → 2880',3:'Residual',6:'attn out [2,2880]',9:'MLPBlock (MoE)',10:'RMSNorm [2,2880]',11:'Router',12:'Linear 2880 → 32',14:'Top-K Expert Selection',16:'Expert weight Softmax [2,4]',17:'Expert MLP',18:'MLP #1',19:'전체 Expert weight [32,5760,2880]',20:'Top-4 선택 [2,4,5760,2880]',22:'SwiGLU',24:'마지막 차원 5760 → 2880 + 2880',26:'MLP #2',30:'Distributed All-Reduce [2,4,2880]',32:'Expert weight Sum',
35:'NO.',37:'Keun Young Co.',38:'year month day ( )',39:'Keun Young Co.',45:'Keun Young Co.',46:'NO.',49:'Keun Young Co.',50:'NO. year month day ( )',51:'- 소프트 맥스 와 교차 엔트로피 오차를 조합한 계층',52:'year month day ( )',53:'Keun Young Co.',
55:'- 에지를 검출해 물체의 경계를 폐곡선으로 나타내는 것.',56:'- 에지 검출 알고리즘은 물체 내부의 명암이 서서히 변해있고 물체',57:'경계는 명암이 급격히 변하는 특성을 활용함',58:'- 에지 연산자:',59:'에지를 검출하기 위해 사용하는 연산 방법 또는 그 연산에 사용되는 필터 (커널)',60:'- 컨볼루션은 계산방법 에지연산자(에지필터)는 그 계산에 넣어주는 숫자 배열.',61:'ex. 간단하게 1차원 영상이 있음',64:'영상위를 한 칸씩 움직이며 컨볼루션을 진행. (겹쳐있는 값끼리 곱하고 전부 더하기)',67:'따라서 첫번째 결과는 0',69:'- 램프 에지 (ramp edge)',70:'명암이 몇 화소에 걸쳐 변하는 현상',71:'(여러 화소에 걸쳐 점진적으로 변하는 에지)',72:'영교차 (zero crossing)',73:'왼쪽과 오른쪽에 부호가 다른 반응이 나타나고 자신은 0을 갖는 위치',74:'- 가장 널리 쓰이는 에지 연산자의 필터 (프레윗 아님 소벨 연산자)',75:'- 에지 검출 : 목적',76:'에지 연산자 : 에지를 검출하기 위한 연산 방법',77:'프레윗 / 소벨 연산자 : 대표적인 에지 연산자',78:'프레윗 / 소벨 필터 : 해당 연산자에서 사용하는 구체적인 필터',79:'Keun Young Co.',82:'Keun Young Co.',87:'Keun Young Co.',91:'작은평수를 구매가능',96:'② 수익환원법 (상가. 오피스텔. 소형 APT)',102:'투자수익률.',108:'월세수익',112:'www.kumhongland',113:'떠나야 할 때를',
115:'과제네',120:'Ragged Array 작성',129:'- makeArray',131:'<문제>',133:'만들어',134:'를',135:'다음 그림과 같은 Ragged Array',136:'행의 합을',137:'- printArray',138:'값을 초기화하고, 출력한 후에 각',140:'구하여 출력하세요.',142:'(length X)',143:'3단계의 작업',144:'만들고 초기화하고 출력',145:'시작',147:'RA 생성 → makeArray()사용',148:'대상 배열',149:'for each element in',150:'4번 반복',151:'initializeArray',154:'RA값 초기화',155:'호출',156:'파라메터 자리',157:'RA 출력',158:'반복',159:'필요',160:'for-each 문',161:'끝',162:'Row data 합 출력',166:'printArray',167:'initializeArray',169:'makeArray',182:'raggedArray[0] = new int[3]',186:'return array',190:'return',193:'for each element in',198:'배열요소 출력',
}
# Existing GT with visibly questionable transcription, canceled writing or ambiguous scope.
# Keep original text and hold; do not silently repair from language knowledge.
HOLD_SOURCE={
'hf-0008':{0:'single GT spans two separately written headings; literal/case uncertain',2:'literal transcription differs from visible mixed-case heading',3:'Taken vs visible Token; independent transcription review required',7:'Rostary/Rostary spelling uncertain',15:'visible function spelling differs from GT',20:'visible mixed-case identifier differs; review'},
'hf-0013':{3:'visible 에이전트 versus GT 에어컨트; do not auto-correct',5:'visible 약한 versus GT 악한; review',15:'crossed-out sticky-note line',17:'crossed-out sticky-note line',20:'visible 샘플 versus GT 삶; review'},
'hf-0033':{17:'handwriting suffix versus GT uncertain'},
'hf-0048':{4:'visible 출력 versus GT 홈럭; review',9:'crossed-out heading'},
'hf-0057':{5:'visible 포함 versus GT 포험; review',10:'overwritten heading; scope uncertain',12:'GT 이용하 versus visible 이용해; review',16:'visible 노름 versus GT 노릅; review'},
'hf-0081':{0:'crossed-out heading',8:'GT phrase differs from visible handwriting',14:'isolated mathematical superscript e',15:'overwritten clause; independent review'},
'hf-0098':{10:'visible 내가 versus GT 네가; review'},
}
PAGE_NOTES={
'hf-0005':'Most body text absent from original2GT. Tensor-only shapes treated structured symbols, prose/code labels included. Tree connectors not characters of prose.',
'hf-0008':'Three structural arrows; first heading combines two detector regions. Several spelling/case discrepancies held. No semantic correction.',
'hf-0013':'Sticky note crosses earlier text; canceled lines held. Agent label added late in source order must precede definition.',
'hf-0029':'Text broadly covered; printed footer/header absent. Rho description split across GTs.',
'hf-0033':'Formula/matrix and top scribbles excluded. Crossed line ignored; prose covers classification/regression.',
'hf-0048':'Formulas excluded; overwritten heading held. Average pooling word 계산 is split by detector.',
'hf-0052':'One body line missing plus printed header; visible reverse-side show-through not target.',
'hf-0057':'Overwritten headings and source typos held; code-like explanatory text retained.',
'hf-0077':'Most edge-detection notes missing; formulas/vector arrays excluded; all visible prose enumerated. Printed header already GT.',
'hf-0081':'Overwritten words, annotation 에서 and line/superscript split; retain ambiguous spans in holds.',
'hf-0087':'Math-bearing prose represented as formula in legacy; retained as mixed math ignore pending separate math scoring. Plain prose included.',
'hf-0088':'Main prose covered. Footer GT is truncated at image edge; no inferred completion.',
'hf-0090':'Formula worksheet; text callouts split into three predicted boxes. Mixed math/prose unresolved; no pure text equation transcription invented.',
'hf-0094':'Large lower-page GT omission. Hard handwriting held individually; yield fraction and numeric scenarios excluded as structured math.',
'hf-0098':'Poem plus author line; missing first short line. No literary correction by known poem.',
'hf-0111':'CSS code; literal GT preserved unless held. Braces are isolated symbols; no semantic code fixes.',
'hf-0137':'Extensive missing prose and flowchart labels. Table digits, arrows and mathematical loop predicates excluded. Flowchart reading order not unique; spatial order is provisional, not human certified.',
}
