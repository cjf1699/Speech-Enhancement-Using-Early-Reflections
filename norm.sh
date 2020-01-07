cd $1
for file in `ls *wav`
do
    newname=$2/$file
    echo "$newname"
    sox $file $newname --norm=-6
done
