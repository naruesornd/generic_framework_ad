import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
import pandas as pd
from data_processor import DataProcessor
from data_processor import CycleProcessor
from feature_engineering import FeatureEngineering
from model.coarse_feature_selection.cfs import random_forest_regressor
from model.fine_feature_selection.ffs import fine_feature_selection
from model.lstm_model.enhanced_lstm import lstm_model
from utils.simulate.simulate_plant import simulate_plant
import pandas as pd
import numpy as np
class FeatureSelectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("feature selection and modeling")
        self.setGeometry(100, 100, 800, 600)
        # initialize variables

        self.dp = None
        self.file_path = ""
        self.all_features = []
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 文件选择区域
        file_layout = QHBoxLayout()
        self.file_label = QLabel("no file selected")
        file_button = QPushButton("select file")
        file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_button)
        main_layout.addLayout(file_layout)
        
        # 特征选择区域
        features_layout = QHBoxLayout()
        
        # 可用特征列表
        available_layout = QVBoxLayout()
        available_layout.addWidget(QLabel("available features:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.ExtendedSelection)
        available_layout.addWidget(self.available_list)
        features_layout.addLayout(available_layout)
        
        # 按钮区域
        buttons_layout = QVBoxLayout()
        buttons_layout.addStretch()
        
        self.to_input_btn = QPushButton(">> input feature")
        self.to_input_btn.clicked.connect(self.add_to_input)
        self.to_input_btn.setEnabled(False)
        buttons_layout.addWidget(self.to_input_btn)
        
        self.to_output_btn = QPushButton(">> output feature")
        self.to_output_btn.clicked.connect(self.add_to_output)
        self.to_output_btn.setEnabled(False)
        buttons_layout.addWidget(self.to_output_btn)
        
        self.clear_input_btn = QPushButton("clear input")
        self.clear_input_btn.clicked.connect(self.clear_input)
        self.clear_input_btn.setEnabled(False)
        buttons_layout.addWidget(self.clear_input_btn)
        
        buttons_layout.addStretch()
        features_layout.addLayout(buttons_layout)
        
        # 输入特征列表
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("input features:"))
        self.input_list = QListWidget()
        self.input_list.setSelectionMode(QListWidget.ExtendedSelection)
        input_layout.addWidget(self.input_list)
        features_layout.addLayout(input_layout)
        
        # 输出特征列表
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("output feature:"))
        self.output_list = QListWidget()
        self.output_list.setSelectionMode(QListWidget.SingleSelection)
        output_layout.addWidget(self.output_list)
        features_layout.addLayout(output_layout)
        
        main_layout.addLayout(features_layout)
        
        # 执行按钮
        self.run_button = QPushButton("modeling")
        self.run_button.clicked.connect(self.run_pipeline)
        self.run_button.setEnabled(False)
        main_layout.addWidget(self.run_button)
        
        # 状态栏
        self.status_bar = self.statusBar()

        # 实时预测模拟按钮
        self.simulate_button = QPushButton("simulate only (no retrain)")
        self.simulate_button.clicked.connect(self.run_simulation_only)
        self.simulate_button.setEnabled(False)
        main_layout.addWidget(self.simulate_button)

    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
        self,
        "select file",
        "",
        "data files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(file_path.split("/")[-1])
            
            try:
                # 初始化DataProcessor
                self.dp = DataProcessor(file_path)
                self.dp.change_pivot('site_date_tz','param_name','display_value')
                self.dp.rename_column_to_timestamp('site_date_tz')
                self.dp.rename_column_to_feedflow('FeedFlowRate')
                self.dp.rename_column_to_permeateflow('PermeateFlowRate')
                self.dp.rename_column_to_feedpressure('PrimaryPressure')
                self.dp.rename_column_to_concentrateflow('ConcentrateFlowRate')
                self.dp.rename_column_to_differentialpressure('MembraneDifferentialPressure')
                self.dp.drop_NA_with_feature(features=['FeedFlow', 'FeedTemperature'])
                # 获取所有特征
                self.all_features = [col for col in self.dp.df.columns if col != 'timestamp']
                self.available_list.clear()
                self.available_list.addItems(self.all_features)
                
                # 启用控件
                self.to_input_btn.setEnabled(True)
                self.to_output_btn.setEnabled(True)
                self.run_button.setEnabled(True)
                self.simulate_button.setEnabled(True)
                self.status_bar.showMessage(f"successfully loaded file {file_path}, total features: {len(self.all_features)} ")

                
            except Exception as e:
                QMessageBox.critical(self, "error", f"file loading failed: {str(e)}")
                self.status_bar.showMessage(f"file loading failed: {str(e)}")
    
    def add_to_input(self):
        selected_items = [item.text() for item in self.available_list.selectedItems()]
        
        for item_text in selected_items:
            # 避免重复添加
            if not self.input_list.findItems(item_text, Qt.MatchExactly):
                self.input_list.addItem(item_text)
            
            # 从可用列表中移除
            list_items = self.available_list.findItems(item_text, Qt.MatchExactly)
            if list_items:
                row = self.available_list.row(list_items[0])
                self.available_list.takeItem(row)
        
        self.clear_input_btn.setEnabled(self.input_list.count() > 0)
    
    def add_to_output(self):
        selected_items = [item.text() for item in self.available_list.selectedItems()]
        
        if selected_items:
            # 只取第一个选中的作为输出特征
            item_text = selected_items[0]
            
            # 清空现有输出特征
            self.output_list.clear()
            self.output_list.addItem(item_text)
            
            # 从可用列表中移除
            list_items = self.available_list.findItems(item_text, Qt.MatchExactly)
            if list_items:
                row = self.available_list.row(list_items[0])
                self.available_list.takeItem(row)
    
    def clear_input(self):
        # 将输入特征移回可用列表
        while self.input_list.count() > 0:
            item = self.input_list.takeItem(0)
            self.available_list.addItem(item.text())
        
        self.clear_input_btn.setEnabled(False)
    
    def run_pipeline(self):
        if self.input_list.count() == 0:
            QMessageBox.warning(self, "warning", "please select at least one input feature")
            return
        
        if self.output_list.count() == 0:
            QMessageBox.warning(self, "warning", "please select one output feature")
            return
        
        # 获取选择的特征
        input_features = [self.input_list.item(i).text() for i in range(self.input_list.count())]
        output_feature = self.output_list.item(0).text()
        
        self.status_bar.showMessage("start processing pipeline...")
        
        try:
            # 1. 周期处理
            cp = CycleProcessor(column_name='FeedFlow', df=self.dp.df, threshold=10)
            cp.identify_cycles()
            cp.assign_cycle_features()
            cp.export_files('../data/cycle_processing_data/factory1.csv')
            
            # 2. 特征工程
            fe = FeatureEngineering(self.dp)
            
            # 生成交叉特征时排除输出特征
            drop_features = ['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure', output_feature]
            fe.generate_cross_features(drop_features=drop_features)
            fe.lag_engineer()
            self.dp.df = fe.df
            
            # 3. 粗粒度特征选择
            # 排除时间戳和不需要的特征
            exclude_features = ['timestamp', 'ConcentratePressure', 'PermeatePressure', 'PermeateFlow','PermeateConductivity', output_feature]
            fs = [f for f in self.dp.df.columns 
                 if f not in exclude_features and f != '']
            print(fs)
            top_k_features = random_forest_regressor(
                self.dp, output_feature, fs, plant_name='plant1'
            )
            
            # 4. 细粒度特征选择
            # s_features = fine_feature_selection(
            #     self.dp, top_k_features, [output_feature]
            # )
            
            # 5. LSTM建模
            lstm_model(self.dp, top_k_features, [output_feature], feature_name=output_feature)
            
            # ✅ 模型训练完成后启动实时预测模拟
            #simulate_plant(top_k_features, output_feature)

            QMessageBox.information(self, "finish", "modeling pipeline executed successfully!")
            self.status_bar.showMessage("modeling pipeline executed successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "error", f"execution failed: {str(e)}")
            self.status_bar.showMessage(f"execution failed: {str(e)}")

    def run_simulation_only(self):
        if self.output_list.count() == 0:
            QMessageBox.warning(self, "warning", "please select one output feature")
            return

        output_feature = self.output_list.item(0).text()

        try:
            # 获取测试数据中的输入特征
            test_data_path = f"../data/model_data/test_data_export_{output_feature}.csv"
            df = pd.read_csv(test_data_path)

            top_k_features = pd.read_csv(f"../data/temp_data/top_k_features_plant1_{output_feature}.csv")
            top_k_features = top_k_features.iloc[:,1].tolist()

            # 启动模拟器（用之前保存的模型）
            simulate_plant(top_k_features, output_feature)

            QMessageBox.information(self, "simulate", "simulation completed successfully!")
            self.status_bar.showMessage("simulation completed successfully")

        except Exception as e:
            QMessageBox.critical(self, "error", f"simulation failed: {str(e)}")
            self.status_bar.showMessage(f"simulation failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureSelectionApp()
    window.show()
    sys.exit(app.exec_())