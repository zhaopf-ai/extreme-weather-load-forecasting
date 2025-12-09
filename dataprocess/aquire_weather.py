from wwo_hist import retrieve_hist_data
import os
os.chdir("D:\python project\image extreme weather\data")
frequency=1
start_date = '1-JAN-2006'
end_date = '4-MAR-2006'
api_key = 'b881eab199f241128d920825250301'
location_list = ['Boston']

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)