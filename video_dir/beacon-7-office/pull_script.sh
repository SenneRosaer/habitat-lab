echo "machine api.wandb.ai" >> /root/.netrc
echo "  login user" >> /root/.netrc
echo "  password a220697dbcaf0bd9df2137876a9b3190835413ad" >> /root/.netrc

wandb pull --entity senne-rosaer --project office-simulator 2maa03dh	
