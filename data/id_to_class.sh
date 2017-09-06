#!/bin/bash


INPUT=$1
OUT_PATH=$2
LABELS=$3

for ID in $(ls $INPUT)
  do
  echo $ID
  label=$(grep $ID $LABELS | awk '{print$2}')
  if [ ! -d "$OUT_PATH/$label" ]; then
    mkdir "$OUT_PATH/$label"
  fi
  for fichero in $(ls $INPUT/$ID | grep jpg)
    do
      cp $INPUT/$ID/$fichero $OUT_PATH/$label/${ID}_${fichero}
  done
done
