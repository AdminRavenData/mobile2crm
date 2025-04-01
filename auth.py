import os
import hashlib
import re

# Path to the secrets file
secrets_file = ".streamlit/secrets.toml"

# Generate the new password hash
password = "Textanalysis@25"  # Password from your login screen
new_hash = hashlib.sha256(password.encode()).hexdigest()

# Read the current file
if os.path.exists(secrets_file):
    with open(secrets_file, 'r') as f:
        content = f.read()
    
    # Use regex to find and replace only the password_hash value
    # This preserves all whitespace, formatting, and other content
    pattern = r'(password_hash\s*=\s*)"([^"]*)"'
    if re.search(pattern, content):
        modified_content = re.sub(pattern, f'\\1"{new_hash}"', content)
        
        # Write the modified content back
        with open(secrets_file, 'w') as f:
            f.write(modified_content)
        
        print(f"Successfully updated only the password_hash in {secrets_file}")
        print(f"New hash for '{password}': {new_hash}")
    else:
        print(f"Couldn't find password_hash entry in {secrets_file}")
else:
    print(f"Error: {secrets_file} does not exist")