tmp=0
while [ $tmp -le 9 ]
do
    echo $tmp
    nohup python3 test.py --dst_label=$tmp >> ./log/$tmp.txt &
    let tmp++
done

# nohup python3 test.py --dst_label=$tmp >> ./log/$tmp.txt &
# echo "$tmp"
# tmp = $((1+tmp))
