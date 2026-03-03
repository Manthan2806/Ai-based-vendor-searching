


        // State variables
        let currentUserType = 'vendor'; // 'vendor' or 'customer'
        let currentAuthMode = 'register'; // 'register' or 'login'?

        function switchUserType(type) {
            currentUserType = type;
            
            // Update Tab Styling
            document.getElementById('tab-vendor').classList.remove('active');
            document.getElementById('tab-customer').classList.remove('active');
            document.getElementById(`tab-${type}`).classList.add('active');

            updateVisibleForm();
        }

        function toggleAuthMode(mode) {
            currentAuthMode = mode;
            
            // Update Header Title
            document.getElementById('main-title').innerText = mode === 'register' ? 'Create an Account' : 'Welcome Back';

            // Show appropriate section
            if(mode === 'register') {
                document.getElementById('register-forms').classList.add('active');
                document.getElementById('login-forms').classList.remove('active');
            } else {
                document.getElementById('login-forms').classList.add('active');
                document.getElementById('register-forms').classList.remove('active');
            }

            updateVisibleForm();
        }

        function updateVisibleForm() {
            // Hide all inner forms first
            document.getElementById('form-vendor-reg').style.display = 'none';
            document.getElementById('form-customer-reg').style.display = 'none';
            document.getElementById('form-vendor-login').style.display = 'none';
            document.getElementById('form-customer-login').style.display = 'none';

            // Construct the ID of the form to show based on state
            const targetFormId = `form-${currentUserType}-${currentAuthMode === 'register' ? 'reg' : 'login'}`;
            document.getElementById(targetFormId).style.display = 'block';
        }
