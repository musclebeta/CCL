import json


file_name1 = r'CCL/CFN2.1/cfn-train.json'
file_name2 = r'CCL/CFN2.1/cfn-test-A.json'
file_name3 = r'CCL/CFN2.1/cfn-dev.json'
file_names = [file_name1, file_name2, file_name3]



for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        
        datas = json.load(file)
        
        
        fe_count = len(datas) 
        count = 0
        
        for data in datas:
        
            cfn_spans = data['cfn_spans']
            text = data['text']
            with open(f'CCL/CFN2.1/{file_name[10:-5]}_cfn_spans.txt', 'a') as file:
                    for i in range(len(cfn_spans)):
                        start = cfn_spans[i]['start']
                        end = cfn_spans[i]['end']
                        fe_name = cfn_spans[i]['fe_name']
                        id = data['sentence_id']
                        target = data['target']
                        
                        span_text = data['text'][start:end+1]
                        if i == 0:
                            file.write(f'句子{id} :' + '论元: 1. ' + f'“{span_text}”' +'【' + str(start) + '-' + str(end) + '】' + f'/{fe_name}/ ')
                        elif i < len(cfn_spans)-1:
                            file.write( f'{i+1}. ' + f'“{span_text}”' +'【' + str(start) + '-' + str(end) + '】'  + f'/{fe_name}/ ')
                        elif i == len(cfn_spans)-1:
                            file.write( f'{i+1}. ' + f'“{span_text}”' +'【' + str(start) + '-' + str(end) + '】'  + f'/{fe_name}/ ' + '\n')
                            for i in range(len(target)):
                            
                                start1 = target[i]['start']
                                end1 = target[i]['end']
                                tar = data['text'][start1:end1+1]
                                if len(target) > 1:
                                    if i == 0:
                                        file.write('目标词: ' + tar )
                                    else:
                                        file.write(' ' + tar + '【' + str(target[0]['start']) + '-' + str(target[0]['end']) + '】' + '【' + str(target[-1]['start']) + '-' + str(target[-1]['end']) + '】' +'\n')
                                        
                                else:
                                    file.write('目标词: ' + tar + '【' + str(start1) + '-' + str(end1) + '】' + '\n')
                            file.write('文本是：' + text +'\n')
        

