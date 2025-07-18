import pandas as pd
import ast

def process_to_geojson(df,colnames=['com']):
    for colname in colnames:
        df[colname+'X']  = df[colname].str[0]
        df[colname+'Y']  = df[colname].str[1]

    #df.drop(colnames, axis=1).to_file(outfile, driver='GeoJSON')  
    return df.drop(colnames, axis=1)

def process_lists_to_geojson(df, colnames=['com']):
    for colname in colnames:
        df[colname + '_str'] = df[colname].apply(
            lambda lst: ','.join(map(str, lst)) 
                        if isinstance(lst, (list, tuple)) and lst is not None 
                        else None
        )
    return df.drop(columns=colnames, errors='ignore')


def convert_to_list(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        else:
            return value.tolist()
    return value

def get_str_list_from_df(df, colnames=['com']):
    for colname in colnames:
        str_col = colname + '_str'

        def parse_comma_list(val):
            if val == '' or pd.isna(val):
                return []
            return [float(item.strip()) for item in val.split(',') if item.strip()]

        df[colname] = df[str_col].apply(parse_comma_list)

    colnames_str = [colname + '_str' for colname in colnames]
    return df.drop(colnames_str, axis=1)

 
def subset_df_to_tuples(df,colnames=['com']):
    for colname in colnames:
        df[colname] = df.apply(lambda row: (row[colname+'X'], row[colname+'Y']), axis=1)
    
    colnamesX=[colname+'X' for colname in colnames]
    colnamesY=[colname+'Y' for colname in colnames]
    
    return df.drop(colnamesX, errors='ignore', axis=1).drop(colnamesY, errors='ignore', axis=1)