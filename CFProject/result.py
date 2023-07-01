import pandas as pd

min = pd.read_csv(r'G:\result\idm_error_min.csv')#, delim_whitespace=True, encoding='utf-8')
pso = pd.read_csv(r'G:\result\idm_error_window_size.csv')
tv = pd.read_csv(r'G:\result\idm_error_tv.csv')
min_gap = pd.read_csv(r'G:\result\idm_error_min_gap.csv')

result = pd.DataFrame(data=[[pso.MSE_V.mean(), pso.MSE_GAP.mean(), pso.RMSE_V.mean(), pso.RMSE_GAP.mean()],
                            [tv.MSE_V.mean(), tv.MSE_GAP.mean(), tv.RMSE_V.mean(), tv.RMSE_GAP.mean()],

                            ],
                     columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
                     index=['PSO_Modified_window_size', 'Time_varying', ])

result = pd.DataFrame(data=[[min.MSE_V.mean(), min.MSE_GAP.mean(), min.RMSE_V.mean(), min.RMSE_GAP.mean()],
                            [min_gap.MSE_V.mean(), min_gap.MSE_GAP.mean(), min_gap.RMSE_V.mean(), min_gap.RMSE_GAP.mean()],
                            [pso.MSE_V.mean(), pso.MSE_GAP.mean(), pso.RMSE_V.mean(), pso.RMSE_GAP.mean()],
                            [tv.MSE_V.mean(), tv.MSE_GAP.mean(), tv.RMSE_V.mean(), tv.RMSE_GAP.mean()],

                            ],
                     columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
                     index=['Min', 'PSO_window_size','PSO_Modified_window_size', 'Time_varying', ])