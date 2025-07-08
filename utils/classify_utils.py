import time
import numpy as np
import pandas as pd 

def TEtxt2df(stitchNode_output_dir):
    """
    Input - directory for stitchNode output
    Ouput - pandas dataframe
    """
    df = pd.read_csv(stitchNode_output_dir,skiprows=1,header=None)

    df.columns = ["TID","year","month","day","hour","nodex","nodey","LON","LAT","MSLP","WS","MSLPCC20",\
                  "MSLPCC55","DEEPSHEAR","UPPTKCC","MIDTKCC","LOWTKCC","Z500CC","VO500AVG","RH100MAX",\
                  "RH850AVG","T850","Z850","ZS","U850DIFF","WS200PMX"]

    df['ISOTIME']=pd.to_datetime(dict(year=df.year,month=df.month,day=df.day,hour=df.hour)) 
    
    new_order = ["TID","LAT","LON","ISOTIME","MSLP","WS","MSLPCC20","MSLPCC55","DEEPSHEAR",\
                 "UPPTKCC","MIDTKCC","LOWTKCC","Z500CC","VO500AVG","RH100MAX","RH850AVG",\
                 "T850","Z850","ZS","U850DIFF","WS200PMX"]

    df_track=df[new_order] 

    return df_track

def cyclone_classifier(dfin):
    """
    Input - pandas dataframe from TEtxt2df()
    Ouput - pandas dataframe with classified cyclones
    """
    print("SyCLoPS main classification program starts ...") ;startt=time.time()
    dfin['mslcc_ratio']=dfin.MSLPCC20/dfin.MSLPCC55
    ## Generate arrays of placeholder labels
    Full_Name=np.array(["Non-labeled"]*len(dfin),dtype=object)
    short_label=np.array(["NLB"]*len(dfin),dtype=object)

    ## High-altitude Branch Labeling
    dfin.Z850=dfin.Z850/9.81 # gravity constant 9.81 m/s^2
    cond_hal=dfin.Z850<dfin.ZS #High-altitude Condition
    df_hatl=dfin[(cond_hal) & ((dfin.MIDTKCC<0)|(dfin.UPPTKCC<0))]; hatl_id=df_hatl.index.values #nodes that satisfy criteria of HATL
    df_hal=dfin[(cond_hal) & ~((dfin.MIDTKCC<0)|(dfin.UPPTKCC<0))]; hal_id=df_hal.index.values #nodes that satisfy criteria of HAL
    Full_Name[hatl_id]="High-altitude Thermal Low"; short_label[hatl_id]="HATHL"
    Full_Name[hal_id]="High-altitude Low"; short_label[hal_id]="HAL"

    ## Dryness Branch Labeling
    cond_dry=dfin.RH850AVG<=60 #Dryness Condition
    cond_cv=(((dfin.VO500AVG>=0) & (dfin.LAT>=0.0)) | ((dfin.VO500AVG<0) & (dfin.LAT<0.0))) #Cyclonic Condition

    df_dsd=dfin[~(cond_hal) & (cond_dry) & ~(dfin.LOWTKCC<0)]; dsd_id=df_dsd.index.values #nodes that satisfy criteria of DSD
    df_dothl=dfin[~(cond_hal) & (cond_dry) & (dfin.LOWTKCC<0) & (cond_cv)]; dothl_id=df_dothl.index.values #nodes that satisfy criteria of DOTHL
    df_thl=dfin[~(cond_hal) & (cond_dry) & (dfin.LOWTKCC<0) & ~(cond_cv)]; thl_id=df_thl.index.values #nodes that satisfy criteria of THL
    
    Full_Name[dsd_id]="Dry Disturbance"; short_label[dsd_id]="DSD"
    Full_Name[dothl_id]="Deep (Orographic) Thermal Low"; short_label[dothl_id]="DOTHL"
    Full_Name[thl_id]="Thermal Low"; short_label[thl_id]="THL"

    ## Tropical Branch Labeling
    cond_trop=(dfin.RH100MAX>20) & (dfin.DEEPSHEAR<13) & (dfin.T850>280) #Tropical Condition
    cond_tc=(dfin.UPPTKCC<-147) & (dfin.LOWTKCC<0) & (dfin.MSLPCC20>225) #Tropical Cyclone Condition
    cond_td=(dfin.MSLPCC55>160) & (dfin.UPPTKCC<0)  #Tropical Depression Condition
    cond_md=(dfin.RH850AVG>85) & (dfin.U850DIFF>0)  #Monsoon System Condition

    df_dst=dfin[~(cond_hal) & ~(cond_dry) & (cond_trop) & ~(cond_cv)]; dst_id=df_dst.index.values #nodes that satisfy criteria of DST
    df_tc=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv)  &(cond_trop) & (cond_tc)]; tc_id=df_tc.index.values #nodes that satisfy criteria of TC
    df_tdmd=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv) & (cond_trop)& ~(cond_tc) & (cond_td) & (cond_md)]
    tdmd_id=df_tdmd.index.values #nodes that satisfy criteria of TD(MD)
    df_tdew=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv) & (cond_trop)& ~(cond_tc) & (cond_td) & ~(cond_md)]
    tdew_id=df_tdew.index.values #nodes that satisfy criteria of TD
    df_tloml=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv) & (cond_trop) & ~(cond_tc) & ~(cond_td) & (cond_md)]
    tloml_id=df_tloml.index.values #nodes that satisfy criteria of TLO(ML)
    df_tloew=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv) & (cond_trop) & ~(cond_tc) & ~(cond_td) & ~(cond_md)]
    tloew_id=df_tloew.index.values #nodes that satisfy criteria of TLO
    df_ms=dfin[~(cond_hal) & ~(cond_dry) & (cond_cv) & (cond_trop) & ~(cond_tc) & (cond_md)]
    
    Full_Name[dst_id]="Tropical Disturbance"; short_label[dst_id]="DST"
    Full_Name[tc_id]="Tropical Cyclone"; short_label[tc_id]="TC"
    Full_Name[tdmd_id]="Tropical Depression(Monsoon Depression)"; short_label[tdmd_id]="TD(MD)"
    Full_Name[tdew_id]="Tropical Depression"; short_label[tdew_id]="TD"
    Full_Name[tloml_id]="Tropical Low (Monsoon Low)"; short_label[tloml_id]="TLO(ML)"
    Full_Name[tloew_id]="Tropical Low"; short_label[tloew_id]="TLO"

    ## Extratropical Branch Labeling
    df_dse=dfin[~(cond_hal) & ~(cond_dry) & ~(cond_trop) & ~(cond_cv)]; dse_id=df_dse.index.values #nodes that satisfy criteria of DSE
    Full_Name[dse_id]="Extratropical Disturbance"; short_label[dse_id]="DSE"
    cond_sc=(dfin.LOWTKCC<0)&(dfin.Z500CC>0)&(dfin.WS200PMX>30) #SC Condition
    df_sc=dfin[~(cond_hal) & ~(cond_dry) & ~(cond_trop) & (cond_cv) & (cond_sc)]; sc_id=df_sc.index.values #nodes that satisfy criteria of SC
    df_ex=dfin[~(cond_hal) & ~(cond_dry) & ~(cond_trop) & (cond_cv) & ~(cond_sc)]; ex_id=df_ex.index.values #nodes that satisfy criteria of EX
    
    Full_Name[sc_id]="Subtropical Cyclone"; short_label[sc_id]="SC"
    Full_Name[ex_id]="Extratropical Cyclone"; short_label[ex_id]="EX"

    #Assign all the labels to the columns:
    dfin['Short_Label']=short_label
    dfin['Full_Name']=Full_Name

    ## Step TWO: Other information about the nodes and tracks
    #Denoting Binary tags for Tropical_Flag and Transition_Zone:
    cond_trans=(cond_trop) & ((dfin.RH100MAX<55) | (dfin.DEEPSHEAR>10)) & (abs(dfin.LAT)>15)
    cond_trans2=(cond_trop) & ((dfin.RH100MAX<55) & (dfin.DEEPSHEAR>10)) & (abs(dfin.LAT)>15)
    trans_flag=np.zeros(len(dfin))
    trop_flag=np.zeros(len(dfin))
    dfin['Transition_Zone']=trans_flag
    dfin['Tropical_Flag']=trop_flag
    dfin.loc[cond_trans,'Transition_Zone']=1
    dfin.loc[cond_trop,'Tropical_Flag']=1

    ## Track labeling
    tctrack=pd.unique(df_tc.TID)[df_tc.groupby('TID')['LON'].count()>=6]
    mstrack=pd.unique(df_ms.TID)[df_ms.groupby('TID')['LON'].count()>=10]
    tctrack_id=dfin[dfin.TID.isin(tctrack)].index.values #TC Track   
    mstrack_id=dfin[dfin.TID.isin(mstrack)].index.values #MS Track
    tctrack_id=dfin[dfin.TID.isin(tctrack)].index.values #TC Track   
    mstrack_id=dfin[dfin.TID.isin(mstrack)].index.values #MS Track

    ## ------------------ Identifying extratropical transition (EXT) ------------------ ##
    ## ------------------ tropical transition (TT) Nodes ------------------ ##
    
    #EXT nodes
    dftc=dfin[dfin.TID.isin(tctrack)]
    extflag=np.zeros(len(tctrack)).astype(int)-1
    for c, i in enumerate(tctrack):
        df0=dftc[dftc.TID==i]
        lst_tc=df0[(df0.Short_Label=='TC')].index[-1]
        dfe=df0.loc[lst_tc+1:]
        dfe=dfe[dfe.Tropical_Flag==0]
        if len(dfe)>1 or (df0.index[-1]-lst_tc==1 and len(dfe)==1):
            fst_ex=dfe.index[0]
            extflag[c]=fst_ex
    extflag=np.delete(extflag, np.argwhere(extflag==-1))
    
    #TT nodes
    ttflag=np.zeros(len(tctrack)).astype(int)-1
    for c, i in enumerate(tctrack):
        df0=dftc[dftc.TID==i]
        if df0.iloc[0].Tropical_Flag==0 or df0.iloc[0].Transition_Zone>0:
            fsttc=df0[(df0.Short_Label=='TC')].index[0]
            ttflag[c]=fsttc
    ttflag= np.delete(ttflag, np.argwhere(ttflag==-1))

    ## Writing information for Track_Info
    track_label=np.array(["Track"]*len(dfin),dtype=object)
    track_label[tctrack_id]=track_label[tctrack_id]+'_TC'
    track_label[mstrack_id]=track_label[mstrack_id]+'_MS'
    track_label[extflag]=track_label[extflag]+'_EXT'
    track_label[ttflag]=track_label[ttflag]+'_TT'
    dfin['Track_Info']=track_label

    ## Output the LPS classified catalog
    dfout=dfin[['TID','LON','LAT','ISOTIME','MSLP','WS','Full_Name','Short_Label','Tropical_Flag','Transition_Zone','Track_Info']]
    dfout.to_parquet("SyCLoPS_classified.parquet") ;endt=time.time()
    print("Time lapsed (s) for the main classification section: "+ str(endt-startt))

    return dfout
