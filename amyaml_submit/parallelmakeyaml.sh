dir="scripts" 

# make *.sh expand to nothing instead of literal when none found
shopt -s nullglob

for f in "$dir"/*.sh; do
  [[ -f "$f" ]] || continue        # skip if not a regular file
  echo "generating yaml for the file $f" 
  
  base="${f##*/}"      # drop path -> "foo.sh"
  name="${base%.sh}"   # drop trailing .sh -> "foo" 

  bash env_add_tmpl.sh $name 4 $base # hardcoding 4 nodes for now 
done 

# clear submit.sh 
: > submit.sh 

for f in "$dir"/*.yaml; do
  [[ -f "$f" ]] || continue        # skip if not a regular file 
  echo "adding yaml to submit $f" 
  
  # printf 'kubectl apply -f amyaml_submit/%s\n' "$f" >> submit.sh 
  printf 'kraken --project-name Obsidian jobs create -i amyaml_submit/%s\n' "$f" >> submit.sh 

done 

chmod +x submit.sh 