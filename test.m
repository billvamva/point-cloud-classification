%[status, result]=system('python3 test_gian.py -p filepath')
clear all
% close all
% clc
addpath('Dataset_generation/')
color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
%m = load_models('uniform');
run("flight_data_generator.m");