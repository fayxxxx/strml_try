import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
pio.templates.default = "none"

#-------------------------------------数据读入---------------------------------------
# 支持文件上传
# 后续可以扩展连接数据库
#-----------------------------------------------------------------------------------

with st.sidebar:
    st.header("Data Uploading")
    # 读入数据
    uploaded_train = st.file_uploader("Please Upload Training Data: ")
    if uploaded_train:
        train = pd.read_csv(uploaded_train)
    else:
        train = None
    
    uploaded_test = st.file_uploader("Please Upload Testing Data: (Optional)")
    if uploaded_test:
        test = pd.read_csv(uploaded_test)
    else:
        test = None
        
    st.header("Data Science Procedure")
    option = st.radio(
    'Please select:',
        ('Data Preview', 'Exploratory Data Analysis', 'Data Manipulation', 'Machine Learning'))

#-------------------------------------原数据预览---------------------------------------
# 数据预览。
# 初步查看数据信息。
#-------------------------------------------------------------------------------------

if option == 'Data Preview':
    if train is not None:
        st.header("Data Preview")
        st.write("-------------------------------------------")
        
        # 展示数据样本
        st.write("**Raw Data**")
        st.write(train)
        
        # 展示数据信息
        col_1_1, col_1_2 = st.columns(2)
        with col_1_1:
            st.write("**Data Infomation**")
            buffer = StringIO() 
            train.info(buf=buffer, verbose=True)
            s = buffer.getvalue() 
            st.text_area("", s, height = 300)
        with col_1_2:
            st.write("**Data Describe**")
            st.write("")
            st.write(train.describe())
    else:
        "Please upload data in the sidebar."

#-------------------------------------探索性数据分析---------------------------------------
# 一元、二元、多元分析
#-----------------------------------------------------------------------------------------

elif option == 'Exploratory Data Analysis':
    st.header("Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["One Dimensional Analysis", "Two Dimensional Analysis", "Multi Dimensional Analysis"])

    # 一元分析
    with tab1:
        col_2_1, col_2_2 = st.columns(2)
        with col_2_1:
            fea_option = st.selectbox('Please Choose One Feature:', train.columns)
        
        with col_2_2:
            if str(train[fea_option].dtype)[:3] == "int" or str(train[fea_option].dtype)[:5] == "float":
                fig_option = st.selectbox('Please Choose One Method:', ['BoxPlot', 'Histogram'])
            
            if train[fea_option].dtype == np.dtype("O"):
                fig_option = st.selectbox('Please Choose One Method:', ['Pie'])
            
        if fig_option == 'BoxPlot':
            fig = px.box(train[fea_option])
            st.plotly_chart(fig)
        
        if fig_option == 'Histogram':
            fig = px.histogram(train[fea_option], histfunc='count')
            st.plotly_chart(fig)

        if fig_option == 'Pie':
            label_dist = train[fea_option].value_counts()
            #st.write(label_dist)
            fig = px.pie(label_dist, names = label_dist.index, values = label_dist)
            st.plotly_chart(fig)

    # 二元分析
    with tab2:
        col_2_3_1, col_2_3_2 = st.columns(2)
        with col_2_3_1:
            fea_x = st.selectbox('Please Select Feature X:', options = train.columns)
        with col_2_3_2:    
            fea_y = st.selectbox('Please Select Feature Y:', options = train.columns)
        
        col_2_4_1, col_2_4_2, col_2_4_3 = st.columns(3)
        with col_2_4_1:
            scatter_size = st.text_input("scatter size")
        with col_2_4_2:
            scatter_color = st.text_input("scatter color")
        with col_2_4_3:
            scatter_symbol = st.text_input("scatter symbol")
        if not scatter_size:
            scatter_size = None
        if not scatter_color:
            scatter_color = None
        if not scatter_symbol:
            scatter_symbol = None
            
        fig = px.scatter(train, fea_x, fea_y, color = scatter_color, symbol = scatter_symbol, size = scatter_size)
        st.plotly_chart(fig)

    # 多元分析
    with tab3:
        fig_option = st.radio('Please Choose One Method:', ['Heat Map'])
        with st.expander('Please Select Features:'):
            feas_option = st.multiselect(' ', options = train.columns, default = list(train.columns))
    
        if fig_option == 'Heat Map':
            sns_colorscale = [[0.0, '#3f7f93'], #cmap = sns.diverging_palette(220, 10, as_cmap = True)
                                [0.071, '#5890a1'],
                                [0.143, '#72a1b0'],
                                [0.214, '#8cb3bf'],
                                [0.286, '#a7c5cf'],
                                [0.357, '#c0d6dd'],
                                [0.429, '#dae8ec'],
                                [0.5, '#f2f2f2'],
                                [0.571, '#f7d7d9'],
                                [0.643, '#f2bcc0'],
                                [0.714, '#eda3a9'],
                                [0.786, '#e8888f'],
                                [0.857, '#e36e76'],
                                [0.929, '#de535e'],
                                [1.0, '#d93a46']]
            tmp = train[feas_option]
            corr = np.array(tmp.corr())
            N = len(feas_option)
            #hovertext = 
            heat = go.Heatmap(z=corr,
                            x=feas_option,
                            y=feas_option,
                            xgap=1, ygap=1,
                            colorscale=sns_colorscale,
                            colorbar_thickness=20,
                            colorbar_ticklen=3,
                            #hovertext=hovertext,
                            #hoverinfo='text'
                            )
            title = 'Correlation Matrix'               

            layout = go.Layout(title_text=title, title_x=0.5, 
                            width=600, height=600,
                            xaxis_showgrid=False,
                            yaxis_showgrid=False,
                            yaxis_autorange='reversed')
            
            fig=go.Figure(data=[heat], layout=layout)        
            st.plotly_chart(fig)

#-------------------------------------数据处理---------------------------------------
# 数据处理
# 支持数据删除
# 缺失值处理
# 改变数据类型
#-------------------------------------------------------------------------------------

elif option == 'Data Manipulation':
    st.header("Data Manipulation")
    tab_dm_1, tab_dm_2, tab_dm_3, tab_dm_4 = st.tabs(["Delete Columns", "Missing Data", "Change Data Types", "Categorical Feature Encoder"])

    # 数据列删除
    with tab_dm_1:
        del_cols = st.multiselect("Please select the columns not used in the training and testing: ", train.columns, default=None)
        if del_cols:
            train = train.drop(columns=del_cols)
            if test is not None:
                test = test.drop(columns=del_cols)

        st.write("***training data preview:***")
        st.write(train.head(5))
        if test is not None:
            st.write("***testing data preview:***")
            st.write(test.head(5))

    # 缺失数据处理
    with tab_dm_2:
        ms_option = st.multiselect("Please select: ", ["Delete missing values", "Fill missing values"], default=None)
        if ms_option:
            if ms_option[0] == "Delete missing values":
                ori_len = len(train)
                train.dropna(inplace=True)
                del_len = len(train)
                st.write("Before deleting, there are ", ori_len, " rows.")
                st.write("Now, there are ", del_len, " rows.")
            elif ms_option[0] == "Fill missing values":
                train_des = train.describe()
                ms_cols = train.isnull().sum()
                ms_cols = list(ms_cols[ms_cols > 0].index)
                with st.expander("fill with mean/median"):
                    flt_col = train.select_dtypes("float").columns
                    ms_flt_col = list(set(flt_col) & set(ms_cols))
                    fill_mean = st.multiselect("(fill with mean value) select columns: ", ms_flt_col, default=ms_flt_col)
                    if fill_mean:
                        guess_fea = train[fill_mean].dropna().mean()
                        guess_fea = guess_fea.to_dict()
                        train = train.fillna(guess_fea)
                        if test is not None:
                            test = test.fillna(guess_fea)
                        st.write("Successfully filled NA with mean value!")
                    
                    int_col = train.select_dtypes("int64").columns
                    ms_int_col = list(set(int_col) & set(ms_cols))
                    fill_median = st.multiselect("(fill with median value) select columns: ", ms_int_col, default=ms_int_col)
                    if fill_median:
                        guess_fea = train[fill_median].dropna().median()
                        guess_fea = guess_fea.to_dict()
                        train = train.fillna(guess_fea)
                        if test is not None:
                            test = test.fillna(guess_fea)
                        st.write("Successfully filled NA with median value!")
              
                with st.expander("self-defining"):
                    for col in ms_cols:
                        if col not in fill_mean and col not in fill_median:
                            fill_val = st.text_input(col)
                            train[col] = train[col].fillna(fill_val)
                            if test is not None:
                                test[col] = test[col].fillna(fill_val)

    # 更改数据类型
    with tab_dm_3:
        col_dm_1, col_dm_2 = st.columns(2)
        with col_dm_2:
            with st.expander("change the data types here"):
                for col in train.columns:
                    change_col = st.multiselect(col + " (" + str(train[col].dtypes) + ")" , ['int64', 'float64', 'object'], default= None)
                    if change_col:
                        train[col] = train[col].astype(change_col[0])
                        if test is not None:
                            test[col] = test[col].astype(change_col[0])
            
        with col_dm_1:
            train_datatype = train.dtypes.reset_index() 
            train_datatype[0] = train_datatype[0].astype(str) 
            train_datatype.columns = ['column', 'datatype']
            st.write("***DataTypes:***")
            st.write(train_datatype)

    # 编码方式选择：
    with tab_dm_4:
        # ------------------------------------------------------
        st.session_state.train_without_encoder = train
        if test is not None:
            st.session_state.test_without_encoder = test
        # ------------------------------------------------------
        
        col_dm_3, col_dm_4 = st.columns(2)
        with col_dm_3:
            encoder_method = st.radio('Encoder Method: ', ['OrdinalEncoder', 'OneHotEncoder'])
        
        with col_dm_4:
            obj_cols = list(train.select_dtypes("object").columns)
            if encoder_method == 'OrdinalEncoder':
                ordinal_encoder_cols = st.multiselect('OrdinalEncoder columns: ', obj_cols, default=obj_cols)
                oec = OrdinalEncoder()
                if test is not None:
                    oec = oec.fit(pd.concat([train[ordinal_encoder_cols], test[ordinal_encoder_cols]]))
                    test[ordinal_encoder_cols] = oec.transform(test[ordinal_encoder_cols])
                else:
                    oec = oec.fit(train[ordinal_encoder_cols])
                train[ordinal_encoder_cols] = oec.transform(train[ordinal_encoder_cols])
            
            if encoder_method == 'OneHotEncoder':
                onehot_encoder_cols = st.multiselect('OneHotEncoder columns: ', obj_cols, default=obj_cols)
                oec = OneHotEncoder()
                if test is not None:
                    tmp = pd.concat([train[onehot_encoder_cols], test[onehot_encoder_cols]]).reset_index(drop=True)
                    train_test_dummy = pd.get_dummies(tmp)
                    train_test_dummy = pd.concat([tmp, train_test_dummy], axis= 1).drop_duplicates()
                    train = pd.merge(train, train_test_dummy, how='left', on = onehot_encoder_cols).drop(columns=onehot_encoder_cols)
                    test = pd.merge(test, train_test_dummy, how='left', on = onehot_encoder_cols).drop(columns=onehot_encoder_cols)
                else:
                    tmp = train[onehot_encoder_cols]
                    train_dummy = pd.get_dummies(tmp)
                    train_dummy = pd.concat([tmp, train_dummy], axis= 1).drop_duplicates()
                    train = pd.merge(train, train_dummy, how='left', on = onehot_encoder_cols).drop(columns=onehot_encoder_cols)
        st.write(train)
    
    st.session_state.train = train
    if test is not None:
        st.session_state.test = test



#-------------------------------------机器学习---------------------------------------
# 数据预览。
# 初步查看数据信息。
#-------------------------------------------------------------------------------------

elif option == "Machine Learning":
    st.header("Do Some Machine Learning !")

    train = st.session_state.train
    if test is not None:
        test = st.session_state.test
    
    col_ml_1, col_ml_2, col_ml_3 = st.columns(3)
    with col_ml_1:
        task = st.selectbox('Task:', ['classification', 'regression'])
    with col_ml_2:
        target = st.selectbox('Target:', train.columns)
    with col_ml_3:
        if task == 'classification':
            n_class = len(train[target].unique())
            if n_class == 2:
                criteria = st.selectbox('Criteria:', ['roc_auc']) # 'accuracy', 'f1'
            elif n_class > 2:
                criteria = st.selectbox('Criteria:', ['accuracy', 'roc_auc_ovr', 'f1_micro'])
        if task == 'regression':
            criteria = st.selectbox('Criteria:', ['neg_mean_squared_error'])

    # ------------------数据准备-----------------------
    y = train[target]
    train = train.drop(columns = [target])
    X = train

    no_encoder_train = st.session_state.train_without_encoder
    obj_cols = list(no_encoder_train.select_dtypes("object").columns)
    no_encoder_train[obj_cols] = no_encoder_train[obj_cols].astype("category")
    no_encoder_train = no_encoder_train.drop(columns = [target])
    # -----------------------------------------------

    tab3_1, tab3_2 = st.tabs(['Training', 'Predicting'])
    if task == 'classification':
        with tab3_1:
            model_performance = {}

            st.write("**Model Comparison:**")
            col_4_1, col_4_2 = st.columns(2)
            with col_4_1:
                cv = st.text_input("Please input the cross-validation fold K: (K>=2)", value=5)
                cv = int(cv)
            with col_4_2:
                if st.button("start comparison"):
                    with st.spinner("Still running..."):
                        for tmp_model in ['Support Vector Machine', 'Gaussian Naive Bayesian', 'Random Forest', 'lightgbm']:
                            if tmp_model == 'lightgbm':
                                lgb_clf = lgb.LGBMClassifier()
                                fit_param = {'categorical_feature' : obj_cols}
                                scores = cross_val_score(lgb_clf, no_encoder_train, y, cv=cv, scoring=criteria, fit_params = fit_param)
                                model_performance[tmp_model] = np.round(scores.mean(),3)
                            
                            if tmp_model == 'Support Vector Machine':
                                svm_clf = SVC(random_state=1128)
                                scores = cross_val_score(svm_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)

                            if tmp_model == 'Gaussian Naive Bayesian':
                                gnb_clf = GaussianNB()
                                scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)
                                
                            if tmp_model == 'Random Forest':
                                rf_clf = RandomForestClassifier()
                                scores = cross_val_score(rf_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)
                        st.session_state.model_performance = model_performance
                    st.success('Done!')

            st.write("**Performance:**")
            try:
                if st.session_state.model_performance:
                    with st.container():
                        model_performance = pd.DataFrame.from_dict(st.session_state.model_performance, orient = 'index', columns = [criteria])
                        st.write(model_performance.T)
            except:
                pass
        
            st.write('**Further prediction:**')
            col_train_1, col_train_2 = st.columns(2)
            with col_train_1:
                model_option = st.selectbox("Choose the model:", [None, 'lgb'])
                #if model_option == 'svm':
                #    final_model = svm_clf.fit(X, y)
                #elif model_option == 'nb':
                #    final_model = gnb_clf.fit(X, y)
                #elif model_option == 'rf':
                #    final_model = rf_clf.fit(X, y)
                if model_option == 'lgb':
                    with col_train_2:
                        if st.button("Hyper params tuning"):
                            with st.spinner('still running...'):
                                from paramtuning import objective
                                import optuna
                                study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
                                func = lambda trial: objective(trial, no_encoder_train, y, obj_cols)
                                study.optimize(func, n_trials=3)
                                st.session_state.study = study
                            st.success('Done!')
                            st.balloons()
            try:
                if st.session_state.study:
                    with st.container():
                        st.write("Best score: ", np.round(study.best_value,3))
                        for key, value in study.best_params.items():
                            st.write(f"\t\t{key}: {value}")
                    final_model = lgb.LGBMClassifier(objective="binary", **st.session_state.study.best_params)
                    X_train, X_test, y_train, y_test = train_test_split(no_encoder_train, y, test_size=0.2, random_state=0)
                    final_model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_test, y_test)],
                        eval_metric="auc",
                        early_stopping_rounds=100,
                        categorical_feature=obj_cols
                    )
                    st.session_state.final_model = final_model
            except:
                pass

        with tab3_2:
            if test is not None:
                if target in test.columns:
                    test = test.drop(columns = [target])
                tmp_test = test
                no_encoder_test = st.session_state.test_without_encoder
                no_encoder_test[obj_cols] = no_encoder_test[obj_cols].astype("category")
                if target in no_encoder_test.columns:
                    no_encoder_test = no_encoder_test.drop(columns = [target])

                if st.button("start prediction"):
                    if model_option == 'lgb':
                        y_pred = st.session_state.final_model.predict(no_encoder_test)
                        y_pred = pd.DataFrame(y_pred, columns = ['prediction'])
                    else:
                        if model_option:
                            y_pred = st.session_state.final_model.predict(tmp_test)
                            y_pred = pd.DataFrame(y_pred, columns = ['prediction'])
                
                    col_show_1, col_show_2 = st.columns(2)
                    with col_show_1:
                        st.write("Predictions: ", y_pred)
                    with col_show_2:
                        @st.cache
                        def convert_df(df):
                            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                            return df.to_csv().encode('utf-8')

                        pred = convert_df(y_pred)
                        st.download_button(
                            label="Download data as CSV",
                            data=pred,
                            file_name='pred.csv',
                        )
            else:
                st.write("***Please Upload Testing Data***")

    if task == 'regression':
        with tab3_1:
            model_performance = {}

            st.write("**Model Comparison:**")
            col_4_1, col_4_2 = st.columns(2)
            with col_4_1:
                cv = st.text_input("Please input the cross-validation fold K: (K>=2)", value=5)
                cv = int(cv)
            with col_4_2:
                if st.button("start comparison"):
                    with st.spinner("Still running..."):
                        # 
                        for tmp_model in ['Linear Regression', 'Lasso', 'DecisionTree', 'GradientBoosting']:

                            if tmp_model == 'Linear Regression':
                                lnr_clf = LinearRegression()
                                scores = cross_val_score(lnr_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)

                            if tmp_model == 'Lasso':
                                from sklearn.linear_model import Lasso
                                lasso_clf = Lasso()
                                scores = cross_val_score(lasso_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)

                            if tmp_model == 'DecisionTree':
                                from sklearn.tree import DecisionTreeRegressor
                                dt_clf=DecisionTreeRegressor()
                                scores = cross_val_score(dt_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)
                            
                            if tmp_model == 'GradientBoosting':
                                from sklearn import ensemble
                                gbt_clf = ensemble.GradientBoostingRegressor()
                                scores = cross_val_score(gbt_clf, X, y, cv=cv, scoring=criteria)
                                model_performance[tmp_model] = np.round(scores.mean(),3)
                            
                        st.session_state.model_performance = model_performance
                    st.success('Done!')

            st.write("**Performance:**")
            try:
                if st.session_state.model_performance:
                    with st.container():
                        model_performance = pd.DataFrame.from_dict(st.session_state.model_performance, orient = 'index', columns = [criteria])
                        st.write(model_performance.T)
            except AttributeError:
                pass
