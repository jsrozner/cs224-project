#!/bin/bash

if [ -d ./data_bk ]; then
  echo "data_bk already exists exitting"
  exit
fi


mv data data_bk

if [ -d ./data ]; then
  echo "data already exists exitting"
  exit
fi

mv data_mod data

