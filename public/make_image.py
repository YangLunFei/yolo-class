import os
import pandas as pd
import numpy as np
import cv2
import datetime


def f_read_emotion_csv(path, jeouls='all'):  # jeouls : 'j01', j02', j03', 'all'
    df = pd.read_csv(path, skipinitialspace=True)

    g, w = [], []
    for i in range(1, 61):
        g.append('g%02d' % i)
        w.append('w%02d' % i)

    # getTime을 1초씩 증가하여 60개 만듦
    df['g01'] = pd.to_datetime(df['getTime'], format='%Y-%m-%d %H:%M:%S')
    for i in range(2, 61):
        df['g%02d' % i] = df['g01'] + datetime.timedelta(seconds=(i - 1))

    if (jeouls == 'j01') or (jeouls == 'all'):
        dfs = df[df['jeoulID'] == 1]
        ar_g = np.array(dfs[g]).reshape(-1, 1)
        ar_w = np.array(dfs[w]).reshape(-1, 1)

        dfa = pd.DataFrame(ar_g, columns=['getTime'])
        dfa['j01'] = ar_w
        # print(dfa.shape)

    if (jeouls == 'j02') or (jeouls == 'all'):
        dfs = df[df['jeoulID'] == 2]
        ar_g = np.array(dfs[g]).reshape(-1, 1)
        ar_w = np.array(dfs[w]).reshape(-1, 1)

        dfb = pd.DataFrame(ar_g, columns=['getTime'])
        dfb['j02'] = ar_w
        # print(dfb.shape)

    if (jeouls == 'j03') or (jeouls == 'all'):
        dfs = df[df['jeoulID'] == 3]
        ar_g = np.array(dfs[g]).reshape(-1, 1)
        ar_w = np.array(dfs[w]).reshape(-1, 1)

        dfc = pd.DataFrame(ar_g, columns=['getTime'])
        dfc['j03'] = ar_w
        # print(dfc.shape)

    if jeouls == 'all':
        df2 = pd.merge(dfa, dfb, on='getTime', how='outer')
        df2 = pd.merge(df2, dfc, on='getTime', how='outer')
        return df2
    elif jeouls == 'j01':
        return dfa
    elif jeouls == 'j02':
        return dfb
    elif jeouls == 'j03':
        return dfc


def f_get_valid_weight(df, jeoul='all', min_value=10, max_value=2500):
    df2 = df.copy()

    mask = (df2[jeoul] >= min_value) & (df2[jeoul] < max_value)
    return df2[mask]

def read_csv_data(fname, jeoul='all'):
    df = f_read_emotion_csv(fname, jeoul)
    if jeoul=='all':
        df21 = f_get_valid_weight(df=df, jeoul='j01')
        df22 = f_get_valid_weight(df=df, jeoul='j02')
        df23 = f_get_valid_weight(df=df, jeoul='j03')

        df2 = pd.concat([df21, df22, df23], axis=0, ignore_index=True)
    else:
        df2 = f_get_valid_weight(df, jeoul)

    # print()
    return df2

def saveImg(img, path, img_name):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
    # print(is_exists)

    cv2.imwrite(path+'/'+img_name, img,[int(cv2.IMWRITE_JPEG_QUALITY),100])


f_path = 'Z:/nasa_old/video/'
f_name = '20211210093325_KF002101_sensorData.csv'
df = read_csv_data(f_path+f_name, jeoul='j01')

df.set_index('getTime', inplace=True)
df['chickens_numbers'] = np.nan
# print(df)
# print(df.index.values.dtype)

v_path = 'Z:/nasa_old/video/Ch1/'
v_name_time_day = os.listdir(v_path)

img_save_path = 'Z:/nasa_old/video/lunfei_img_make_0724/'

for day in v_name_time_day:
    print(day+' doing!')
    video_names = os.listdir(v_path+day)
    video_num = 0
    for v_n in video_names:
        video_num += 1

        v_p = v_path + day + '/' + v_n

        # CD.load_data(v_p)
        # os.system('cls')
        print("\033c", end="")
        cap = cv2.VideoCapture(v_p)

        time_second = v_p.split('.')[0][-14:]
        # print(time_second)
        star_time = pd.Period(time_second, freq='S')
        # print(star_time)
        # second = CD.second
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(round(fps, 0))
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        second = int(frames / fps)

        end_time = star_time + second
        # print(end_time)
        # print(type(end_time))

        time_list_s_e = pd.to_datetime([str(star_time), str(end_time)], format='%Y-%m-%d %H:%M:%S')

        df_temp = df[time_list_s_e[0]:time_list_s_e[1]]

        if len(df_temp) != 0:
            # print(v_p + ': do it now! {0}/{1}'.format(video_num, len(video_names)))

            time_list = df_temp.index
            time_list_temp = time_list.to_period(freq='S')
            # print(time_list_temp)
            detcet_time = (time_list_temp - star_time)


            # print(detcet_time)
            time_l_s = [t.strftime('%Y%m%d%H%M%S') for t in time_list]
            time_list_temp = [int(str(t).split(' * ')[0][1:]) if str(t) != '<Second>' else 1 for t in detcet_time]

            f = -1
            while True:
                success, frame = cap.read()
                if not success:
                    break

                f += 1
                if f % fps != 0:
                    continue
                times_second = int(f / fps)
                if times_second not in time_list_temp:
                    continue

                try:

                    img_name = time_l_s[time_list_temp.index(times_second)] + '.jpg'
                    # print(img_name)

                    frame = cv2.resize(frame, (int(frame.shape[1]*0.6), int(frame.shape[0]*0.6)))

                    # print(frame.shape)

                    saveImg(frame, img_save_path+day, img_name)
                except:
                    pass

                # cv2.imshow('01', frame)
                # if cv2.waitKey(30) == ord('q'):
                #     exit()
                if times_second == time_list_temp[-1]:
                    break





            # print(time_list_temp)

            # result = CD.detcet_img(time_list_temp, None, view_img=False, save_img=False, dtype='video')
