import openpyxl
import  numpy as np
import pandas as pd

wb = openpyxl.load_workbook(r'C:\Users\TK\Desktop\30.0.xlsx')  # 得到工作簿
print(wb.sheetnames)


ava_list = []
for sheet in wb.sheetnames:
    sheet_num = []
    if sheet in ['EN', 'SF', 'SD', 'PSNR', 'MSE', 'MI', 'VIF', 'AG', 'CC', 'SCD', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']:
        sheet = wb.get_sheet_by_name(sheet)
        prj = sheet.columns
        prjTuple = tuple(prj)
        for cell in prjTuple[0]:
            if cell.value != 'Ours' and cell.value != '=AVERAGE(A2:A22)':
                sheet_num.append(float(cell.value))
        sheet_num = np.array(sheet_num)
        ava = np.mean(sheet_num)
        ava_list.append(ava)
print(ava_list)

ava_list = np.array(ava_list)
data = pd.DataFrame(ava_list, columns=['描述'])

filename = r'C:\Users\TK\Desktop\test.xlsx'
sheetname = 'test1'
wb1 = openpyxl.load_workbook(filename)  # 得到工作簿

if 'test1' in wb1.sheetnames:   # sheet_1是工作表名，可以改
    df_old = pd.DataFrame(pd.read_excel(filename, sheet_name=sheetname))  # 读取原数据文件和表
    print(df_old.shape)
    df3 = pd.concat([data, df_old], axis=1)
    column_old = df_old.shape[1]  # 获取原数据的行数

    book = openpyxl.load_workbook(filename)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    writer.book = book

    df3.to_excel(writer, sheet_name='test1',float_format='%.6f', index=False)
    writer.save()
    writer.close()
else:
    book = openpyxl.load_workbook(filename)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    writer.book = book
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer, sheet_name=sheetname, float_format='%.6f', index=False)
    writer.save()
    writer.close()











