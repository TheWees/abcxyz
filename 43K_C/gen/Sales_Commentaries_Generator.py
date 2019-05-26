from __future__ import unicode_literals, print_function, division

import pandas as pd
import numpy as np
import datetime
import time
import gc

# to make this notebook's output stable across runs
np.random.seed(42)

#avoid scienticfic notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

async def generate_sales_comments(file_name):
    data_file = pd.ExcelFile(file_name)
    sales_data = pd.read_excel(data_file, data_file.sheet_names[0])
    gl_data = pd.read_excel(data_file, data_file.sheet_names[-1])
    sales_data.columns = sales_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    sales_data['period'] = sales_data.period.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    sales_data.sort_values('period',inplace=True)
    sales_data.reset_index(inplace=True, drop=True)
    country = sales_data.country.unique().tolist()
    franchiselevel2 = sales_data.franchiselevel2.unique().tolist()
    sku = sales_data.sku.unique().tolist()
    period = sales_data.period.unique().tolist()
    fbp = ['nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may','nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
       'nts_may_ytd','nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
       'nts_may_qtd']
    ju = ['nts_jun','nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
       'nts_sep_ytd', 'nts_oct_ytd','nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
       'nts_sep_qtd', 'nts_oct_qtd']
    nu = ['nts_nov', 'nts_dec','nts_nov_qtd', 'nts_dec_qtd','nts_nov_ytd', 'nts_dec_ytd']

    period_name = ['nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may', 'nts_jun',
                   'nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_nov', 'nts_dec',
                   'nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
                   'nts_may_ytd', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
                   'nts_sep_ytd', 'nts_oct_ytd', 'nts_nov_ytd', 'nts_dec_ytd',
                   'nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
                   'nts_may_qtd', 'nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
                   'nts_sep_qtd', 'nts_oct_qtd', 'nts_nov_qtd', 'nts_dec_qtd']

    cols = ['country','franchiselevel2','period',
            'nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may', 'nts_jun',
            'nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_nov', 'nts_dec',
            'nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
            'nts_may_ytd', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
            'nts_sep_ytd', 'nts_oct_ytd', 'nts_nov_ytd', 'nts_dec_ytd',
            'nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
            'nts_may_qtd', 'nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
            'nts_sep_qtd', 'nts_oct_qtd', 'nts_nov_qtd', 'nts_dec_qtd']

    cols_sku = ['sku','period',
                'nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may', 'nts_jun',
                'nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_nov', 'nts_dec',
                'nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
                'nts_may_ytd', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
                'nts_sep_ytd', 'nts_oct_ytd', 'nts_nov_ytd', 'nts_dec_ytd',
                'nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
                'nts_may_qtd', 'nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
                'nts_sep_qtd', 'nts_oct_qtd', 'nts_nov_qtd', 'nts_dec_qtd']
    
    program_starts = time.time()
    franchise_status = []
    for c in country:
        for f in franchiselevel2:
            level1_sum = sales_data[cols][(sales_data.country==c) & 
                                       (sales_data.franchiselevel2==f)]\
                                       .groupby(by=['country','franchiselevel2','period']).sum()

            level1_sum.reset_index(inplace=True)
            df = level1_sum.transpose().iloc[2:,:]
            df.columns = df.loc['period']
            df.drop(index='period',inplace=True) 
            if len(df.columns) > 2:
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            for mon in df.index:
                if mon in fbp:
                    if period[1] in df.columns and period[2] in df.columns:
                        if type(df.loc[mon,period[1]]) != type(None) and type(df.loc[mon,period[2]]) != type(None):
                            #comp = df.loc[mon,period[1]] > df.loc[mon,period[2]]
                            if df.loc[mon,period[1]] > df.loc[mon,period[2]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'above', 
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[2]]})
                            elif df.loc[mon,period[1]] < df.loc[mon,period[2]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'below', 
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[2]]})
                elif mon in ju:
                    if period[1] in df.columns and period[2] in df.columns:
                        if type(df.loc[mon,period[1]]) != type(None) and type(df.loc[mon,period[3]]) != type(None):
                            #comp = df.loc[mon,period[1]] > df.loc[mon,period[3]]
                            if df.loc[mon,period[1]] > df.loc[mon,period[3]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'above',
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[3]]})
                            elif df.loc[mon,period[1]] < df.loc[mon,period[3]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'below',
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[3]]})
                else:
                    if period[1] in df.columns and period[2] in df.columns:
                        if type(df.loc[mon,period[1]]) != type(None) and type(df.loc[mon,period[4]]) != type(None):
                            #comp = df.loc[mon,period[1]] > df.loc[mon,period[4]]
                            if df.loc[mon,period[1]] > df.loc[mon,period[4]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'above',
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[4]]})
                            elif df.loc[mon,period[1]] < df.loc[mon,period[4]]:
                                franchise_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                         'result': 'below',
                                                         'difference': df.loc[mon,period[1]]-df.loc[mon,period[4]]})

    print("It has been {0} seconds since the loop started".format(time.time() - program_starts))                    
    print(len(franchise_status))
    
    copy_frame = sales_data[['country','franchiselevel2', 'sku',
   'nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may', 'nts_jun',
   'nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_nov', 'nts_dec',
   'nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
   'nts_may_ytd', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
   'nts_sep_ytd', 'nts_oct_ytd', 'nts_nov_ytd', 'nts_dec_ytd',
   'nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
   'nts_may_qtd', 'nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
   'nts_sep_qtd', 'nts_oct_qtd', 'nts_nov_qtd', 'nts_dec_qtd']][:1]
    
    df = pd.DataFrame().reindex_like(copy_frame)

    def get_percentage(x, period, cols):
        per = 0
        if len(x)>=2:
            if x['index'] in fbp:
                if period[1] in cols and period[2] in cols:
                    if x[period[2]] != 0:
                        per = ((x[period[1]]-x[period[2]])/x[period[2]])*100
                    else:
                        per = 0
            elif x['index'] in ju:
                if period[1] in cols and period[3] in cols:
                    if x[period[3]] != 0:
                        per = ((x[period[1]]-x[period[3]])/x[period[3]])*100
                    else:
                        per = 0
            else:
                if period[1] in cols and period[4] in cols:
                    if x[period[4]] != 0:
                        per = ((x[period[1]]-x[period[4]])/x[period[4]])*100
                    else:
                        per = 0
        return per
    
    idx = pd.IndexSlice
    program_starts = time.time()
    for c in country:
        for f in franchiselevel2:
            level3 = sales_data[cols_sku][(sales_data.country==c) & 
                              (sales_data.franchiselevel2==f)].set_index(['sku','period'])
            df2 = level3.transpose()
            for s in level3.index.levels[0]:
                perf_df = df2.loc[:,idx[s]]
                perf_df.reset_index(inplace=True)
                perf_df['percentage_change'] = perf_df.apply(lambda x: get_percentage(x, period, perf_df.columns), axis=1)
                perf_df = perf_df[['index','percentage_change']].transpose()
                perf_df.reset_index(inplace=True)
                perf_df.drop('period', inplace=True, axis=1)
                perf_df.columns = perf_df.iloc[0]
                perf_df.drop(0, inplace=True ,axis=0)
                perf_df.insert(0,'country', c) 
                perf_df.insert(1,'franchiselevel2', f) 
                perf_df.insert(2,'sku', s) 
                df = pd.concat([df,perf_df], axis=0)
    print("It has been {0} seconds since the loop started".format(time.time() - program_starts)) 
    level3_status = df[1:]
    
    level1_status = pd.DataFrame(franchise_status, columns=['country','franchiselevel2','period_name','result','difference'])
    status = level1_status.merge(level3_status)
    status = status.drop_duplicates()
    print(len(status))
    
    del df, perf_df, level3, level1_status, level3_status, franchise_status, level1_sum
    gc.collect()
    
    def make_list(x,lst=[]):
        if x['result'] == 'above':
            if x[6] > 5:
                lst.append((' '.join(x['sku'].split()[2:]), x[6]))
        elif x['result'] == 'below':                             
            if x[6] < -5:
                lst.append((' '.join(x['sku'].split()[2:]), x[6]))
        return lst    

    final_status = pd.DataFrame(columns=['country','franchiselevel2','period_name', 'difference','result','change_list'])
    program_starts = time.time()
    for c in country:
        for f in franchiselevel2:
            for p in period_name:
                lst = []
                #print('processing', c ,f ,p)
                dff = status[['country','franchiselevel2','period_name',
                              'result','sku','difference',p]][(status.country==c) 
                                           & (status.franchiselevel2==f) 
                                           & (status.period_name==p)]
                if len(dff)>0:
                    dff['change_list'] = dff.apply(lambda x: make_list(x,lst), axis=1) 
                    dff = dff[['country','franchiselevel2','period_name','difference','result','change_list']].iloc[[0]]
                    final_status = pd.concat([final_status,dff], axis=0)
    print("It has been {0} seconds since the loop started".format(time.time() - program_starts)) 
    final_status.reset_index(inplace=True, drop=True)
    final_status['change_list'] = final_status.change_list.apply(lambda x: sorted(x, key = lambda kv: kv[1], reverse=True))
   
    def create_comment(x, text=''):
        lst = x['change_list']
        if len(lst)>0:
            if len(x['period_name'])>7:
                typ = x['period_name'][8:].upper()
            else:
                typ = 'MTD'
            if x['result'] == 'above':
                if x['period_name'] in fbp:       
                    if len(lst)>=3:
                        text = '{} above FBP {} driven by strong performances of {}, {}, {} which sold more than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} above FBP {} driven by strong performances of {}, {} which sold more than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} above FBP {} driven by strong performance of {} which sold more than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])
                if x['period_name'] in ju:
                    if len(lst)>=3:
                        text = '{} above JU {} driven by strong performances of {}, {}, {} which sold more than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} above JU {} driven by strong performances of {}, {} which sold more than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} above JU {} driven by strong performance of {} which sold more than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])
                if x['period_name'] in nu:
                    if len(lst)>=3:
                        text = '{} above NU {} driven by strong performances of {}, {}, {} which sold more than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} above NU {} driven by strong performances of {}, {} which sold more than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} above NU {} driven by strong performance of {} which sold more than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])

            if x['result'] == 'below':
                if x['period_name'] in fbp:       
                    if len(lst)>=3:
                        text = '{} below FBP {} driven by weak sales of {}, {}, {} which sold less than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} below FBP {} driven by weak sales of {}, {} which sold less than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} below FBP {} driven by weak sale of {} which sold less than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])
                if x['period_name'] in ju:
                    if len(lst)>=3:
                        text = '{} below JU {} driven by weak sales of {}, {}, {} which sold less than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} below JU {} driven by weak sales of {}, {} which sold less than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} below JU {} driven by weak sale of {} which sold less than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])
                if x['period_name'] in nu:
                    if len(lst)>=3:
                        text = '{} below NU {} driven by weak sales of {}, {}, {} which sold less than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} below NU {} driven by weak sales of {}, {} which sold less than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} below NU {} driven by weak sale of {} which sold less than {:.2f}% forecasted'.format(x['franchiselevel2'],typ, lst[0][0], lst[0][1])
        return text    
    
    def version(x):
        version = ''
        if x['period_name'] in fbp:
            version = period[1].upper() + ' vs ' + period[2].upper()
        if x['period_name'] in ju:
            version = period[1].upper() + ' vs ' + period[3].upper()
        if x['period_name'] in nu:
            version = period[1].upper() + ' vs ' + period[4].upper()
        return version
    
    final_status['comment'] = final_status.apply(lambda x: create_comment(x), axis=1)
    final_status['period_type'] = final_status.period_name.apply(lambda x: x[8:].upper() if len(x)>7 else 'MTD')
    final_status['version'] = final_status.apply(lambda x: version(x), axis=1)
    final_status.drop('change_list', inplace=True,axis=1)
    
    del status, dff
    gc.collect()
    
    def actuals_comments(sales_data=sales_data, period_name=period_name, 
                         cols=cols, period = period, cols_sku = cols_sku):
        sales_data = sales_data[sales_data.period.isin([period[0],period[1]])]
        print(sales_data.period.value_counts())
        
        program_starts = time.time()
        franchise_act_status = []
        for c in country:
            for f in franchiselevel2:
                level1_act_sum = sales_data[cols][(sales_data.country==c) & 
                                           (sales_data.franchiselevel2==f)]\
                                           .groupby(by=['country','franchiselevel2','period']).sum()

                level1_act_sum.reset_index(inplace=True)
                df = level1_act_sum.transpose().iloc[2:,:]
                df.columns = df.loc['period']
                df.drop(index='period',inplace=True) 
                if len(df.columns) > 2:
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
                for mon in df.index:
                    if period[0] in df.columns and period[1] in df.columns:
                        if type(df.loc[mon,period[0]]) != type(None) and type(df.loc[mon,period[1]]) != type(None):
                            if df.loc[mon, period[1]] > df.loc[mon, period[0]]:
                                act_vs = 'above'
                            elif df.loc[mon, period[1]] < df.loc[mon, period[0]]: 
                                act_vs = 'below'
                        franchise_act_status.append({'country':c,'franchiselevel2':f,'period_name':mon, 
                                                             'result': act_vs, 
                                                             'difference': df.loc[mon,period[1]]-df.loc[mon,period[0]]})
        print("It has been {0} seconds since the loop started".format(time.time() - program_starts))                    
        print(len(franchise_act_status))
        
        level1_act_status = pd.DataFrame(franchise_act_status, 
                                 columns=['country','franchiselevel2','period_name','result','difference'])
        
        copy_frame = sales_data[['country','franchiselevel2', 'sku',
                                 'nts_jan', 'nts_feb', 'nts_mar', 'nts_apr', 'nts_may', 'nts_jun',
                                 'nts_jul', 'nts_aug', 'nts_sep', 'nts_oct', 'nts_nov', 'nts_dec',
                                 'nts_jan_ytd', 'nts_feb_ytd', 'nts_mar_ytd', 'nts_apr_ytd',
                                 'nts_may_ytd', 'nts_jun_ytd', 'nts_jul_ytd', 'nts_aug_ytd',
                                 'nts_sep_ytd', 'nts_oct_ytd', 'nts_nov_ytd', 'nts_dec_ytd',
                                 'nts_jan_qtd', 'nts_feb_qtd', 'nts_mar_qtd', 'nts_apr_qtd',
                                 'nts_may_qtd', 'nts_jun_qtd', 'nts_jul_qtd', 'nts_aug_qtd',
                                 'nts_sep_qtd', 'nts_oct_qtd', 'nts_nov_qtd', 'nts_dec_qtd']][:1]

        df = pd.DataFrame().reindex_like(copy_frame)


        def get_percentage_actuals(x, period, cols):
            per = 0
            if len(x)>=2:    
                if period[1] in cols and period[0] in cols:
                    if x[period[0]] != 0:
                        per = ((x[period[1]]-x[period[0]])/x[period[0]])*100
                    else:
                        per = 0
            return per

        idx = pd.IndexSlice
        program_starts = time.time()
        for c in country:
            for f in franchiselevel2:
                level3 = sales_data[cols_sku][(sales_data.country==c) & 
                                  (sales_data.franchiselevel2==f)].set_index(['sku','period'])
                df2 = level3.transpose()
                for s in level3.index.levels[0]:
                    perf_df = df2.loc[:,idx[s]]
                    perf_df.reset_index(inplace=True)
                    perf_df['percentage_change'] = perf_df.apply(lambda x: get_percentage_actuals(x, period, perf_df.columns), 
                                                                 axis=1)
                    perf_df = perf_df[['index','percentage_change']].transpose()
                    perf_df.reset_index(inplace=True)
                    perf_df.drop('period', inplace=True, axis=1)
                    perf_df.columns = perf_df.iloc[0]
                    perf_df.drop(0, inplace=True ,axis=0)
                    perf_df.insert(0,'country', c) 
                    perf_df.insert(1,'franchiselevel2', f) 
                    perf_df.insert(2,'sku', s) 
                    df = pd.concat([df,perf_df], axis=0)
        print("It has been {0} seconds since the loop started".format(time.time() - program_starts)) 
        level3_act_status = df[1:]
        status_act = level1_act_status.merge(level3_act_status)
        status_act =  status_act.drop_duplicates()
        print(len(status_act))

        del df, perf_df, level3, level1_act_status, level3_act_status, franchise_act_status, level1_act_sum
        gc.collect()

        def make_list_act(x,lst=[]):
            if x['result'] == 'above':
                if x[6] > 5:
                    lst.append((' '.join(x['sku'].split()[2:]), x[6]))
            elif x['result'] == 'below':                             
                if x[6] < -5:
                    lst.append((' '.join(x['sku'].split()[2:]), x[6]))
            return lst   

        final_act_status = pd.DataFrame( columns=['country','franchiselevel2','period_name', 'difference','result','change_list']) 

        program_starts = time.time()
        for c in country:
            for f in franchiselevel2:
                for p in period_name:
                    lst = []
                    dff = status_act[['country','franchiselevel2','period_name',
                                  'result','sku','difference',p]][(status_act.country==c) 
                                               & (status_act.franchiselevel2==f) 
                                               & (status_act.period_name==p)]
                    if len(dff)>0:
                        dff['change_list'] = dff.apply(lambda x: make_list_act(x,lst), axis=1) 
                        dff = dff[['country','franchiselevel2','period_name','difference','result','change_list']].iloc[[0]]
                        final_act_status = pd.concat([final_act_status,dff], axis=0)
        print("It has been {0} seconds since the loop started".format(time.time() - program_starts)) 
        final_act_status.reset_index(inplace=True, drop=True)
        final_act_status['change_list'] = final_act_status.change_list.apply(lambda x: sorted(x, key = lambda kv: kv[1], reverse=True))

        def create_comment_actuals(x, text=''):
            lst = x['change_list']
            if len(lst)>0:
                if len(x['period_name'])>7:
                    typ = x['period_name'][8:].upper()
                else:
                    typ = 'MTD'
                if x['result'] == 'above':
                    if len(lst)>=3:
                        text = '{} above {} {} driven by strong performances of {}, {}, {} which sold more than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} above {} {} driven by strong performances of {}, {} which sold more than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} above {} {} driven by strong performance of {} which sold more than {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[0][1])

                if x['result'] == 'below':                 
                    if len(lst)>=3:
                        text = '{} below {} {} driven by weak sales of {}, {}, {} which sold less than {:.2f}%, {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[1][0], lst[2][0], lst[0][1], lst[1][1], lst[2][1])
                    if len(lst)==2:
                        text = '{} below {} {} driven by weak sales of {}, {} which sold less than {:.2f}%, {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[1][0], lst[0][1], lst[1][1])
                    if len(lst)==1:
                        text = '{} below {} {} driven by weak sale of {} which sold less than {:.2f}% forecasted'.format(x['franchiselevel2'], period[0], typ, lst[0][0], lst[0][1])
            return text    
            
        final_act_status['comment'] = final_act_status.apply(lambda x: create_comment_actuals(x), axis=1)
        final_act_status['period_type'] = final_act_status.period_name.apply(lambda x: x[8:].upper() if len(x)>7 else 'MTD')
        final_act_status['version'] = final_act_status.apply(lambda x: period[1].upper() + ' vs ' + period[0].upper(), axis=1)
        final_act_status.drop('change_list', inplace=True,axis=1)

        return final_act_status
    act_status = actuals_comments()
    final_status = pd.concat([final_status,act_status], axis=0)
    final_status['comment'][final_status.comment==''] = 'Variance of sales actual to predicted/previous year actuals is below 5% so nothing major to comment'
    final_status.reset_index(inplace=True, drop=True)
    final_status = final_status[['country','period_name','period_type','franchiselevel2','version','result','difference','comment']]
    
    del act_status
    gc.collect()
    
    def create_gl_comments(sales_data=sales_data, gl_data=gl_data, country=country, franchiselevel2=franchiselevel2):
        sales_data_gl = sales_data[['franchiselevel2', 'sku']]
        sales_data_gl.columns = ['franchiselevel2', 'product']
        gl_data.columns = gl_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        gl_data['version'] = gl_data.version.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        gl_data.sort_values('version',inplace=True)
        gl_data.reset_index(inplace=True, drop=True)
        gl_data.columns = ['country', 'version', 'year', 'product', 'gl_account', 'jan', 'feb', 'mar',
                           'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        gl_data = gl_data[gl_data.gl_account.isin(['Total Gross Profit', 'Brand Marketing Expenses', '650200 - Transportation',
                                           '605140 - Inventory Adjustments EB','650100 - Stock & Shipping',
                                           '605000 - COG&S Sold Trade EB'])]
        gl_data = gl_data.merge(sales_data_gl, on='product', how='left',copy=False)
        gl_data = gl_data.drop_duplicates()
        gl_data.reset_index(inplace=True, drop=True)
        gl_data = gl_data[['country', 'version', 'year', 'franchiselevel2', 'product', 'gl_account', 'jan', 'feb',
               'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
        gl_data['country'] = gl_data.country.apply(lambda x: x[9:])
        account = gl_data['gl_account'].unique().tolist() 
        
        fbp_gl = ['jan', 'feb','mar', 'apr', 'may']
        ju_gl = ['jun', 'jul', 'aug', 'sep', 'oct']
        nu_gl = ['nov', 'dec']
        gl_data['period'] = gl_data.apply(lambda x: x['version'] + '_' + str(x['year']), axis=1)
        gl_data = gl_data[['country', 'franchiselevel2', 'product', 'gl_account', 'period', 'jan', 'feb',
                           'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
        period_gl = gl_data.period.unique()
        
        good = ['605000 - COG&S Sold Trade EB', 'Total Gross Profit', '650100 - Stock & Shipping']
        bad = ['650200 - Transportation','Brand Marketing Expenses','605140 - Inventory Adjustments EB']
        def make_gl_comments(x, a, f, cols):
            text =''
            if period_gl[0] in cols and period_gl[1] in cols:
                if x['period_name'] in fbp_gl:
                    if x[period_gl[0]] > x[period_gl[1]]:
                        if a in good:
                            text = '{} {} is more than {} MTD by {:.2f} indicates steady performance of {}.  '.format(period_gl[0].upper(), a, period_gl[1].upper(), (x[period_gl[0]] - x[period_gl[1]])*1000, f)
                        if a in bad:
                            text = '{} {} is more than {} MTD by {:.2f}, we need to check and report what is driving all these expenses for {}.  '.format(period_gl[0].upper(), a, period_gl[1].upper(), (x[period_gl[0]] - x[period_gl[1]])*1000, f)
                    elif  x[period_gl[0]] < x[period_gl[1]]:
                        if a in good:
                            text = '{} {} is less than {} MTD by {:.2f}, means slow growth, we need to check for further information to check what went wrong for {}.  '.format(period_gl[0].upper(), a, period_gl[1], (x[period_gl[0]] - x[period_gl[1]])*1000, f)
                        if a in bad:
                            text = '{} {} is less than {} MTD by {:.2f}, so, our expenses are under what had been forecasted and we are on track for {}.  '.format(period_gl[0].upper(), a, period_gl[1].upper(), (x[period_gl[0]] - x[period_gl[1]])*1000, f)
            if period_gl[0] in cols and period_gl[2] in cols:   
                if x['period_name'] in ju_gl:
                    if x[period_gl[0]] > x[period_gl[2]]:
                        if a in good:
                            text = '{} {} is more than {} MTD by {:.2f} indicates steady performance of {}.  '.format(period_gl[0].upper(), a, period_gl[2].upper(), (x[period_gl[0]] - x[period_gl[2]])*1000, f)
                        if a in bad:
                            text = '{} {} is more than {} MTD by {:.2f}, we need to check and report what is driving all these expenses for {}.  '.format(period_gl[0].upper(), a, period_gl[2].upper(), (x[period_gl[0]] - x[period_gl[2]])*1000, f)
                    elif  x[period_gl[0]] < x[period_gl[2]]:
                        if a in good:
                            text = '{} {} is less than {} MTD by {:.2f}, means slow growth, we need to check for further information to check what went wrong for {}.  '.format(period_gl[0].upper(), a, period_gl[2].upper(), (x[period_gl[0]] - x[period_gl[2]])*1000, f)
                        if a in bad:
                            text = '{} {} is less than {} MTD by {:.2f}, so, our expenses are under what had been forecasted and we are on track for {}.  '.format(period_gl[0].upper(), a, period_gl[1].upper(), (x[period_gl[0]] - x[period_gl[2]])*1000, f)
            if period_gl[0] in cols and period_gl[3] in cols:
                if x['period_name'] in nu_gl:
                    if x[period_gl[0]] > x[period_gl[3]]:
                        if a in good:
                            text = '{} {} is more than {} MTD by {:.2f} indicates steady performance of {}.  '.format(period_gl[0].upper(), a, period_gl[3].upper(), (x[period_gl[0]] - x[period_gl[3]])*1000, f)
                        if a in bad:
                            text = '{} {} is more than {} MTD by {:.2f}, we need to check and report what is driving all these expenses for {}.  '.format(period_gl[0].upper(), a, period_gl[3].upper(), (x[period_gl[0]] - x[period_gl[3]])*1000, f)
                    elif  x[period_gl[0]] < x[period_gl[3]]:
                        if a in good:
                            text = '{} {} is less than {} MTD by {:.2f}, means slow growth, we need to check for further information to check what went wrong for {}.  '.format(period_gl[0].upper(), a, period_gl[3].upper(), (x[period_gl[0]] - x[period_gl[3]])*1000, f)
                        if a in bad:
                            text = '{} {} is less than {} MTD by {:.2f}, so, our expenses are under what had been forecasted and we are on track for {}.  '.format(period_gl[0].upper(), a, period_gl[3].upper(), (x[period_gl[0]] - x[period_gl[3]])*1000, f)
            return text
        gl_status = pd.DataFrame(columns=['country', 'franchiselevel2', 'period_name', 'comment'])
        program_starts = time.time()
        for c in country:
            for f in franchiselevel2:
                for a in account:
                    level1_sum = gl_data[(gl_data.country==c) & 
                                               (gl_data.franchiselevel2==f) & (gl_data.gl_account==a)]\
                                               .groupby(by=['country','franchiselevel2','gl_account','period']).sum()

                    level1_sum.reset_index(inplace=True)
                    df = level1_sum.transpose().iloc[3:,:]
                    df.columns = df.loc['period']
                    df.drop(index='period',inplace=True) 
                    df.reset_index(inplace=True)
                    df.rename(columns={'index':'period_name'}, inplace=True)
                    df['comment'] = df.apply(lambda x: make_gl_comments(x,a,f, df.columns), axis=1)
                    df.insert(0,'country',c)
                    df.insert(1, 'franchiselevel2', f)
                    df = df[['country', 'franchiselevel2', 'period_name', 'comment']]
                    gl_status = pd.concat([gl_status, df], axis=0)
        print("It has been {0} seconds since the loop started".format(time.time() - program_starts))  
        gl_status.drop_duplicates(inplace=True)
        gl_status['period_name'] = gl_status.period_name.apply(lambda x: 'nts_'+x)
        gl_status = gl_status.groupby(by=['country','franchiselevel2','period_name']).sum()
        gl_status = gl_status[gl_status.comment!='']
        return gl_status
    gl_comments = create_gl_comments()
    
    del sales_data, gl_data
    gc.collect()
    
    final_status_act_vs_act_variance = final_status[(final_status.version== period[1].upper() +' vs '+ period[0].upper()) |
                                      (final_status.comment.str.startswith('Variance of sales actual to predicted'))]
    final_status_non_act_non_variance = final_status[final_status.version!= period[1].upper() +' vs '+ period[0].upper()]
    final_status_non_act_non_variance = final_status_non_act_non_variance[final_status.comment != 'Variance of sales actual to predicted/previous year actuals is below 5% so nothing major to comment' ]
    final_status_non_act_non_variance = final_status_non_act_non_variance.merge(gl_comments, left_on=['country','franchiselevel2','period_name'], 
                                                                                right_on=['country','franchiselevel2','period_name'], how='left')
    final_status_non_act_non_variance.fillna(' ', inplace=True)
    final_status_non_act_non_variance['comment'] = final_status_non_act_non_variance.apply(lambda x: str(x['comment_x']) + '.  '+ str(x['comment_y']), axis=1)
    final_status_non_act_non_variance.drop(['comment_x','comment_y'], inplace=True, axis=1)
    final_status = pd.concat([final_status_non_act_non_variance,final_status_act_vs_act_variance], axis=0)
    print("^^^Yay generated comments!")

    filename = "generated_comments" + "_" + str(datetime.datetime.now().timestamp()) + ".xlsx"
    final_status.to_excel("processed/{}".format(filename))
    # print(filename)
    return filename 