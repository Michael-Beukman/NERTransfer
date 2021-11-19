cat swa/train.txt > current.txt
#jjfor i in */train.txt; do
for j in swa wol kin pcm; do
	i=$j/train.txt
	if [ "$i" == "amh/train.txt" ]; then
		echo "BAD"; continue;
	fi
	if [ "$i" == "yor/train.txt" ]; then
		echo "BAD"; continue;
	fi
	comm -12 <(sort $i) <(sort current.txt) | grep -v ' O' | grep -v -e '^[[:space:]]*$' > tmp.txt
	cat tmp.txt > current.txt
done
