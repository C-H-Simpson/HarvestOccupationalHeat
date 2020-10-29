echo "Enter openID"
read openID
export openID=$openID

echo "Enter openID password"
echo "This will be stored as an environment variable"
read -s openID_password
export openID_password=$openID_password
