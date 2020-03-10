#!/bin/bash

if [ -d ./data_mod ]; then
  echo "data_mod already exists exitting"
  exit
fi

mv data data_mod

if [ -d ./data ]; then
  echo "data already exists exitting"
  exit
fi

mv data_bk data

