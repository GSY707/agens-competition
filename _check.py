import json
s=json.load(open('results/summary.json',encoding='utf-8'))
print('summary.json OK:')
for k,v in s.items():
    print('  %s: %-12s conf=%s' % (k, v['status'], v['confidence']))
fd=json.load(open('results/final_diagnosis_data.json',encoding='utf-8'))
print()
print('final_diagnosis_data.json 特征频率 (BSF 应为 68.6~68.9, 确认公式已修正):')
for k,v in fd.items():
    print('  %s: BSF=%s  2xfr=%s  BPFO=%s  BPFI=%s  FTF=%s' % (
        k, v['features']['bsf'], v['features']['shaft2x'],
        v['features']['bpfo'], v['features']['bpfi'], v['features']['ftf']))
print()
print('各样本主导特征 lead_max (DE 通道):')
for k in ['A','B','C','D']:
    pf=fd[k]['per_feature']
    print('  %s: BPFO=%7.1f  BPFI=%7.1f  BSF=%7.1f  FTF=%7.1f' % (
        k, pf['bpfo']['lead_max'], pf['bpfi']['lead_max'],
        pf['bsf']['lead_max'], pf['ftf']['lead_max']))
