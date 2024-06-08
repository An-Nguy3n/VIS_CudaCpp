import numpy as np
import pandas as pd
import argparse
import os


class Base:
    def __init__(self,
                 data: pd.DataFrame,
                 interest,
                 valuearg_threshold):
        data = data.reset_index(drop=True)
        data.fillna(0.0, inplace=True)

        # Check cac cot bat buoc
        drop_cols = ["TIME", "PROFIT", "SYMBOL", "VALUEARG"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f"Thieu cot {col}")

        # Check dtype cua TIME, PROFIT va VALUEARG
        if data["TIME"].dtype != "int64":
            raise Exception("TIME's dtype must be int64")
        if data["PROFIT"].dtype != "float64":
            raise Exception("PROFIT's dtype must be float64")
        if data["VALUEARG"].dtype not in ["int64", "float64"]:
            raise Exception("VALUEARG's dtype must be int64 or float64")

        # Check thu tu cot TIME va min PROFIT
        if data["TIME"].diff().max() > 0:
            raise Exception("Cot TIME phai giam dan")
        if data["PROFIT"].min() < 0.0:
            raise Exception("PROFIT < 0.0")

        # INDEX
        index = []
        for i in range(data["TIME"].max(), data["TIME"].min()-1, -1):
            if i not in  data["TIME"].values:
                raise Exception(f"Thieu chu ky {i}")

            index.append(data[data["TIME"]==i].index[0])

        index.append(data.shape[0])
        self.INDEX = np.array(index)

        # Loai cac cot co kieu du lieu khong phai int64 va float64
        for col in data.columns:
            if col not in drop_cols and data[col].dtype not in ["int64", "float64"]:
                drop_cols.append(col)

        self.drop_cols = drop_cols
        print("Cac cot khong duoc coi la bien:", self.drop_cols)

        # Attrs
        self.data = data
        self.INTEREST = interest
        self.PROFIT = np.array(data["PROFIT"], float)
        self.PROFIT[self.PROFIT < 5e-324] = 5e-324
        self.VALUEARG = np.array(data["VALUEARG"], float)
        self.BOOL_ARG = self.VALUEARG >= valuearg_threshold

        symbol_name = data["SYMBOL"].unique()
        self.symbol_name = {symbol_name[i]:i for i in range(len(symbol_name))}
        self.SYMBOL = np.array([self.symbol_name[s] for s in data["SYMBOL"]])
        self.symbol_name = {v:k for k,v in self.symbol_name.items()}

        operand_data = data.drop(columns=drop_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = np.transpose(np.array(operand_data, float))

        self.PROFIT_RANK = np.zeros(data.shape[0])
        self.PROFIT_RANK_NI = np.zeros(self.INDEX.shape[0]-1)
        temp_serie = pd.Series([self.INTEREST])
        for i in range(self.INDEX.shape[0]-1):
            start, end = self.INDEX[i], self.INDEX[i+1]
            temp_ = pd.concat([data.loc[start:end-1, "PROFIT"], temp_serie], ignore_index=True)
            temp_rank = np.array(temp_.rank(method="min"), float) / (end-start+1)
            self.PROFIT_RANK[start:end] = temp_rank[:-1]
            self.PROFIT_RANK_NI[i] = temp_rank[-1]


def to_bin_file(base, filename, dtype):
    array = getattr(base, filename)
    with open(f"InputData/{filename}.bin", "wb") as f:
        f.write(np.array(array.shape, np.int32).tobytes())
        f.write(np.array(array, dtype).tobytes())


if __name__ == "__main__":
    os.makedirs("InputData/", exist_ok=True)
    for file in os.listdir("InputData/"):
        os.remove(f"InputData/{file}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath")
    parser.add_argument("--interest")
    parser.add_argument("--valueargThreshold")

    args = parser.parse_args()
    data = pd.read_excel(args.dataPath)
    interest = float(args.interest)
    valuearg_threshold = float(args.valueargThreshold)

    base = Base(data, interest, valuearg_threshold)
    to_bin_file(base, "INDEX", np.int32)
    to_bin_file(base, "PROFIT", np.float64)
    to_bin_file(base, "SYMBOL", np.int32)
    to_bin_file(base, "BOOL_ARG", np.int32)
    to_bin_file(base, "OPERAND", np.float64)
