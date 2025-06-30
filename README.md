**1. Install Python**
```bash
sudo apt install python3 python3-pip python3-venv python3-dev -y
```

**2. Install CLI**
```bash
git clone https://github.com/octra-labs/octra_pre_client.git
cd octra_pre_client

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp wallet.json.example wallet.json
```

**2. Add wallet to CLI**
```bash
nano wallet.json
```
* Replace following values:
  "{
  "wallets": [
    {
      "name": "Wallet1",
      "priv": "private_key_1",
      "addr": "octA1234567890123456789012345678901234567890",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet2",
      "priv": "private_key_2",
      "addr": "octB9876543210987654321098765432109876543210",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet3",
      "priv": "private_key_3",
      "addr": "octC1234567890123456789012345678901234567890",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet4",
      "priv": "private_key_4",
      "addr": "octD9876543210987654321098765432109876543210",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet5",
      "priv": "private_key_5",
      "addr": "octE1234567890123456789012345678901234567890",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet6",
      "priv": "private_key_6",
      "addr": "octF9876543210987654321098765432109876543210",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet7",
      "priv": "private_key_7",
      "addr": "octG1234567890123456789012345678901234567890",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet8",
      "priv": "private_key_8",
      "addr": "octH9876543210987654321098765432109876543210",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet9",
      "priv": "private_key_9",
      "addr": "octI1234567890123456789012345678901234567890",
      "rpc": "https://octra.network"
    },
    {
      "name": "Wallet10",
      "priv": "private_key_10",
      "addr": "octJ9876543210987654321098765432109876543210",
      "rpc": "https://octra.network"
    }
  ]

**add wallet.txt**
addres per lane
**3. Start CLI**
```bash
python3 -m venv venv
source venv/bin/activate
python3 cli.py
```
* This should open a Testnet UI

![image](https://github.com/user-attachments/assets/0ba1d536-4048-4899-a977-4517b2e522cd)


**4. Send transactions**
* Send transactions to my address: `octBvPDeFCaAZtfr3SBr7Jn6nnWnUuCfAZfgCmaqswV8YR5`
* Use [Octra Explorer](https://octrascan.io/) to find more octra addresses


**5. Use Alternative Script**
* If you have issue with official script, I just refined the script with optimizated UI, you can replace the current one with mine by executing this command:
```bash
curl -o cli.py https://raw.githubusercontent.com/0xmoei/octra/refs/heads/main/cli.py
```

**6. Share Feedback**

Always share your feedback about the week's task in discord

---

## Wait for next steps
Stay tuned.
