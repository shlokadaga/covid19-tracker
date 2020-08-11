import sqlite3
from sqlite3 import Error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from covid import Covid
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import datetime
global final_df

today=datetime.date.today()

def sql_connect():
    try:
        conn=sqlite3.connect('covidtracker.db')
        return conn
    except Error:
        print('Could not connect to the database')

def create_table(conn):
    cursor=conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS cases(Date VARCHAR,Confirmed INTEGER, Recovered INTEGER, Deceased INTEGER)')
    conn.commit()

def insert_data(conn):
    cursor = conn.cursor()

    print('Enter the Date')
    date= input()
    print('Enter the number of Confirmed cases')
    confirm = int(input())
    print('Enter the number of Recovered cases')
    recovered = int(input())
    print('Enter the number of Deceased Cases')
    deceased = int(input())
    cursor.execute('INSERT INTO cases(Date,Confirmed,Recovered,Deceased)VALUES(?,?,?,?)',(date,confirm,recovered,deceased))
    conn.commit()

def view_data(conn):
    cursor = conn.cursor()

    ab=cursor.execute('SELECT * FROM cases')

    df=pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
    df=df.rename(columns={'Daily Confirmed':'Confirmed_Cases','Daily Recovered':'Recovered_Cases','Daily Deceased':'Deceased_Cases'})
    df['Active_Cases']=df['Confirmed_Cases']-(df['Recovered_Cases']+df['Deceased_Cases'])
    print('\n')
    abc=df.iloc[:,[0,1,3,5,7]]

    print('\nEnter the TYPE OF TRACKER you want to see\n\n1. LINE GRAPH using MATPLOTLIB\n2. BAR GRAPH\n3. LINE GRAPH using PLOTLY\n4. STATE Wise Data\n5. DONUT CHART')
    a=int(input())
    if a==1:
        plt.style.use('ggplot')

        xdate = ['21 April', '4 May', '18 May', '1 June', '1 July']
        color1 = ['#B6B10D', '#32A8A4', '#298315', '#CF0FCC', '#B20B4D']
        lockdown = ['Lockdown-2', 'Lockdown-3', 'Lockdown-4', 'Lockdown-5', 'Lockdown-6']


        plt.plot(df.Date,df.Confirmed_Cases,linewidth=3,linestyle='-',marker='o',color='red',label='Confirmed Cases')
        plt.plot(df.Date, df.Recovered_Cases, linewidth=3,linestyle='-', marker='o', color='green',label='Recovered Cases')
        plt.plot(df.Date, df.Deceased_Cases,linewidth=3, linestyle='-', marker='o', color='black',label='Deceased Cases')
        plt.title('COVID-19 Tracker')

        for x_plot, c, ld in zip(xdate, color1, lockdown):
            plt.axvline(x=x_plot, label=ld,linestyle='-', color=c,linewidth=7.0,alpha=0.20)

        plt.legend(loc='center')
        plt.xlabel('Date')
        plt.ylabel('Number of Cases')
        plt.xticks(df.Date[::6])
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.show()

    elif a==2:
        fig=make_subplots(rows=2,cols=2,shared_xaxes=True)

        data1=go.Bar(x=df['Date'],y=df['Confirmed_Cases'],name='Confirmed Cases')
        data2=go.Bar(x=df['Date'],y=df['Recovered_Cases'],name='Recovered Cases')
        data3=go.Bar(x=df['Date'],y=df['Active_Cases'],name='Active Cases')
        data4=go.Bar(x=df['Date'],y=df['Deceased_Cases'],name='Deceased Cases')

        fig.add_trace(data3, 1, 1)
        fig.add_trace(data1, 1, 2)
        fig.add_trace(data2, 2, 1)
        fig.add_trace(data4, 2, 2)
        fig.update_layout(title_text='Daily COVID-19 Tracker')
        fig.show()

    elif a==3:
        graph1 = go.Scatter(x=df['Date'],y=df['Confirmed_Cases'],name='Confirmed Cases',mode='markers+lines')
        graph2 = go.Scatter(x=df['Date'], y=df['Recovered_Cases'], name='Recovered Cases',mode='markers+lines')
        graph3 = go.Scatter(x=df['Date'], y=df['Deceased_Cases'], name='Deceased Cases',mode='markers+lines')
        data=[graph3,graph1,graph2]
        layout=go.Layout(title='COVID-19 Cases Tracker')
        figure=go.Figure(data=data,layout=layout)
        figure.update_layout(
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor':'center',
                'yanchor':'top'
            }
        )
        figure.show()

    elif a==4:
        state_data()

    elif a==5:
        colors = ['grey', 'navyblue', 'greenyellow']
        names=['Deceased_Cases','Active Cases','Recovered Cases']
        pie_graph=go.Pie(values=[df['Deceased_Cases'].sum(),df['Active_Cases'].sum(),df['Recovered_Cases'].sum()],labels=names,
                         hole=.2)
        layout=go.Layout(title='COVID-19 Donut chart',)
        figure=go.Figure(data=pie_graph,layout=layout)
        figure.update_traces(textinfo='percent+label',textfont_size=18,marker=dict(colors=colors,line=dict(color='#000000',width=4)))
        figure.show()

def daily_date_wise_data():
    df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
    df = df.rename(columns={'Daily Confirmed': 'Confirmed_Cases', 'Daily Recovered': 'Recovered_Cases',
                            'Daily Deceased': 'Deceased_Cases'})

    print('\n')
    abc = df.iloc[:, [0, 1, 3, 5]]
    print(abc.tail(110))


def view_lockdown_phase_wise_data(conn):
    plt.style.use('ggplot')
    cursor = conn.cursor()
    a = cursor.execute('SELECT * FROM cases')

    df = pd.DataFrame(a, columns=['Date', 'Confirmed_Cases', 'Recovered_Cases', 'Deceased_Cases'])
    df['Active_Cases'] = df['Confirmed_Cases'] - (df['Recovered_Cases'] + df['Deceased_Cases'])
    print(df)
    lkdwn2 = df.iloc[1:13,:]
    lkdwn3 = df.iloc[14:27,:]
    lkdwn4 = df.iloc[28:41,:]
    lkdwn5 = df.iloc[42:71,:]
    lkdwn6 = df.iloc[72:102, :]
    lkdwn7 = df.iloc[103:]
    lockdwn=[lkdwn2,lkdwn3,lkdwn4,lkdwn5,lkdwn6,lkdwn7]
    lkdown = ['LOCKDOWN-2', 'LOCKDOWN-3', 'LOCKDOWN-4', 'LOCKDOWN-5', 'LOCKDOWN-6','LOCKDOWN-7']
    for abc,lkd in zip(lockdwn,lkdown):
        print(lkd)
        print('CONFIRMED CASES : ',abc['Confirmed_Cases'].sum())
        print('RECOVERED CASES : ',abc['Recovered_Cases'].sum())
        print('ACTIVE CASES    : ',abc['Active_Cases'].sum())
        print('DECEASED CASES  : ',abc['Deceased_Cases'].sum())
        print('\n')




    df_lockdown=pd.DataFrame(index=lkdown)
    cc=[]
    rcc=[]
    ac=[]
    dc=[]
    for ab,lk in zip(lockdwn,lkdown):
        cc.append(ab['Confirmed_Cases'].sum())
        rcc.append(ab['Recovered_Cases'].sum())
        dc.append(ab['Deceased_Cases'].sum())
        ac.append(ab['Active_Cases'].sum())

    df_lockdown['Confirmed Cases'] = cc
    df_lockdown['Recovered Cases'] = rcc
    df_lockdown['Deceased Cases'] = dc
    df_lockdown['Active Cases'] = ac

    df_lockdown.plot(kind='barh',y=['Confirmed Cases','Recovered Cases','Deceased Cases','Active Cases'])

    plt.gca().invert_yaxis()

    plt.show()

def case_predict(conn):
    cursor = conn.cursor()
    a = cursor.execute('SELECT * FROM cases')
    df = pd.DataFrame(a, columns=['Date', 'Confirmed_Cases', 'Recovered_Cases', 'Deceased_Cases'])
    total_data=len(df['Confirmed_Cases'])-1
    print('Total Data till now: {}'.format(total_data))
    abc=[n for n in range(total_data)]
    x=np.array(abc).reshape(-1,1)
    rows=df[0:-1]
    rows_column=rows.iloc[:,1]
    y=rows_column.tolist()
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
    model=LinearRegression()
    model.fit(xtrain,ytrain)
    future_value=np.array(x).reshape(-1,1)
    ypred=model.predict(future_value)
    print('Predicted :',ypred)


def state_data():

    state_df = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    state_df['Date'] = state_df.Date.apply(lambda x: pd.to_datetime(x).strftime('%d-%m-%Y'))
    state_df.rename(columns={'AN': 'Andaman Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam',
                             'BR': 'Bihar', 'CH': 'Chandigarh', 'CT': 'Chattisgarh', 'DD': 'Daman & Diu', 'DL': 'Delhi',
                             'GA': 'Goa', 'GJ': 'Gujarat',
                             'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JK': 'Jammu And Kashmir', 'KA': 'Karnataka',
                             'KL': 'Kerela',
                             'LA': 'Ladakh', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur',
                             'ML': 'Meghalaya',
                             'MZ': 'Mizoram', 'NL': 'Nagaland', 'OR': 'Orissa', 'PY': 'Puducherry', 'PB': 'Punjab',
                             'RJ': 'Rajasthan',
                             'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TG': 'Telegana', 'TR': 'Tripura',
                             'UP': 'Uttar Pradesh',
                             'UT': 'Uttarakhand',
                             'WB': 'West Bengal', 'TT': 'Total'}, inplace=True)

    top_affected=state_df.loc[state_df['Status']=='Confirmed']
    top_affected=top_affected.iloc[:,3:-1]
    top_affected=top_affected.reset_index()
    top_affected.drop(['index'],axis=1,inplace=True)
    confirmed=top_affected.sum(axis=0)
    confirmed_df=pd.DataFrame(confirmed,columns=['Confirmed Cases'])

    recovered_state=state_df.loc[state_df['Status']=='Recovered']
    recovered_state=recovered_state.iloc[:,3:-1]
    recovered_state=recovered_state.reset_index()
    recovered_state.drop(['index'],axis=1,inplace=True)
    recovered=recovered_state.sum(axis=0)
    recovered_df=pd.DataFrame(recovered,columns=['Recovered Cases'])

    final_part1=confirmed_df.merge(recovered_df,left_index=True,right_index=True)

    deceased_state = state_df.loc[state_df['Status'] == 'Deceased']
    deceased_state = deceased_state.iloc[:, 3:-1]
    deceased_state = deceased_state.reset_index()
    deceased_state.drop(['index'], axis=1, inplace=True)
    deceased = deceased_state.sum(axis=0)
    deceased_df = pd.DataFrame(deceased, columns=['Deceased Cases'])



    final_df=final_part1.merge(deceased_df,left_index=True,right_index=True)
    final_df['Active Cases']=final_df['Confirmed Cases']-(final_df['Recovered Cases']+final_df['Deceased Cases'])
    final_df.index.name='State'
    fig=make_subplots(rows=2,cols=2,shared_xaxes=True)
    barchart1=go.Bar(x=final_df.index,y=final_df['Confirmed Cases'],name='Confirmed Cases')
    barchart2=go.Bar(x=final_df.index,y=final_df['Recovered Cases'],name='Recovered Cases')
    barchart3=go.Bar(x=final_df.index,y=final_df['Deceased Cases'],name='Deceased Cases')
    barchart4 = go.Bar(x=final_df.index, y=final_df['Active Cases'], name='Active Cases')
    layout=go.Layout(title='Total Confirmed Cases in Each State')
    fig.add_trace(barchart4,1,1)
    fig.add_trace(barchart1, 1, 2)
    fig.add_trace(barchart2, 2, 1)
    fig.add_trace(barchart3, 2, 2)
    fig.update_layout( title_text="COVID-19 case details State wise")
    fig.show()
    population = pd.read_excel('Indian States Population and Area.xlsx')
    df_population = pd.merge( final_df,population, on='State')
    df_population.sort_values(by='Confirmed Cases',ascending=False,inplace=True)
    print(df_population)

def view_date_wise():
    df = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
    df = df.iloc[:, 0:4]
    df.set_index('State', inplace=True)
    pd.options.display.float_format = '{:,.1f}'.format
    df['Active'] = df['Confirmed'] - (df['Recovered'] + df['Deaths'])
    df['Active_Ratio'] = (df['Active'] / (df['Active'] + df['Recovered'] + df['Deaths'])) * 100
    df['Recovered_Ratio'] = (df['Recovered'] / (df['Active'] + df['Recovered'] + df['Deaths'])) * 100
    df['Deceased_Ratio'] = (df['Deaths'] / (df['Active'] + df['Recovered'] + df['Deaths'])) * 100
    df = df.replace(np.nan, 0)
    print(df)
    print()
    state_df = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    state_df['Date'] = state_df.Date.apply(lambda x: pd.to_datetime(x).strftime('%d-%m-%Y'))
    state_df.rename(
        columns={'AN': 'Andaman Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam',
                 'BR': 'Bihar', 'CH': 'Chandigarh', 'CT': 'Chattisgarh', 'DD': 'Daman & Diu', 'DL': 'Delhi',
                 'GA': 'Goa', 'GJ': 'Gujarat',
                 'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JK': 'Jammu And Kashmir', 'KA': 'Karnataka',
                 'KL': 'Kerela',
                 'LA': 'Ladakh', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
                 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OR': 'Orissa', 'PY': 'Puducherry', 'PB': 'Punjab',
                 'RJ': 'Rajasthan',
                 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TG': 'Telegana', 'TR': 'Tripura', 'UP': 'Uttar Pradesh',
                 'UT': 'Uttarakhand',
                 'WB': 'West Bengal', 'TT': 'India'}, inplace=True)

    print('Enter the Date in dd-mm-yyyy format')
    date1=input()
    print('Enter the Name of the State')
    state=input()
    date_particular=state_df.loc[state_df['Date']==date1]
    state_particular=date_particular.loc[:,['Status',state,'India']]
    state_particular.reset_index(inplace=True)
    state_particular.drop('index',inplace=True,axis=1)
    print(state_particular)

def world_data(col,n):
    world_data_df = pd.read_csv(
        'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/worldometer_data.csv')
    world_data_df.rename(columns={'TotalCases':'Confirmed','TotalDeaths':'Deceased','TotalRecovered':'Recovered'},inplace=True)
    world_data_df=world_data_df.replace('',np.nan).fillna(0)
    figure=px.bar(world_data_df.sort_values(by=col).tail(n),x=col,y='Country/Region',orientation='h',color='Continent',
                      text=col,width=1200,color_discrete_sequence=px.colors.qualitative.Dark2)

    figure.update_layout(title='Top 15 Country with most '+col+' Cases in World',xaxis_title='',yaxis_title=''
                         ,yaxis_categoryorder='total ascending')
    figure.show()



conn=sql_connect()
create_table(conn)
covid=Covid()
india_cases=covid.get_status_by_country_name('india')

print('INDIA COVID-19 Cases')
print('CONFIRMED CASES : ',india_cases['confirmed'])
print('ACTIVE CASES    : ',india_cases['active'])
print('RECOVERED CASES : ',india_cases['recovered'])
print('DECEASED CASES  : ',india_cases['deaths'])
print()
print('Hello\nWhat action do you want to perform on this site')
print('1. Enter CORONAVIRUS Data\n2. VIEW Current CORONAVIRUS Case Details\n3. VIEW Lockdown Wise Details \n4. VIEW State Wise Details  \n5. Predict Coronavirus \n6. WORLD data through BAR GRAPH \n7. VIEW Date wise \n8. VIEW State DATA Graphically \n\n')
print('Enter your Choice')
choice=int(input())
if choice==1:
    print('Enter password to enter the DATA')
    password=int(input())
    pawo=734
    if password==pawo:
        insert_data(conn)
elif choice==2:
    view_data(conn)

elif choice==3:
    view_lockdown_phase_wise_data(conn)

elif choice==4:
    view_date_wise()

elif choice==5:
    case_predict(conn)

elif choice==6:
    world_data('Deceased',15)

elif choice==7:
    daily_date_wise_data()

elif choice==8:
    state_data()

else:
    print("Your Choice didn't match")

