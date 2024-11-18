from datetime import datetime, timedelta
def date_now():
    today_date_str=input("请输入今天的日期(example:2022,09,09)：")
    num_people=int(input("请问你要查询几个人："))
    today_date = datetime.strptime(today_date_str, "%Y,%m,%d")
    return today_date,num_people

def date_input(num_people):
    lists = []
    for i in range(0,num_people):
        num_shot=int(input("请输入已经接种了几针："))
        shot_date_str=input("请输入最近一次的接种日期：(example：2022,09,09)：")
        shot_date=datetime.strptime(shot_date_str,"%Y,%m,%d")
        lists.append((num_people,shot_date))
    return lists

def date_output(lists,today_date):
    out_lists=[]
    for num_shot,shot_date in lists:
        dictionary = {}
        if num_shot==0:
            dictionary[True]=today_date
        elif num_shot==1:
            # 存疑
            next_shot_date = shot_date + timedelta(30)
            if today_date>next_shot_date:
                dictionary[True]=today_date
            else:
                dictionary[False]=shot_date+timedelta(30)
        elif num_shot==2:
            next_shot_date=shot_date + timedelta(180)
            if today_date >=next_shot_date :
                dictionary[True] = today_date
            else:
                dictionary[False] = shot_date+timedelta(180)
        elif num_shot==3:
            dictionary[False] =" "
        else:
            print("输入数据有误")
        out_lists.append(dictionary)
    print(out_lists)

today_date,num_people=date_now()
lists=date_input(num_people)
date_output(lists,today_date)
