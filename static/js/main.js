/**
 * LoanPredict AI - Frontend Logic & Interactivity
 */

document.addEventListener('DOMContentLoaded', () => {

    // ---------------------------------------------------------
    // 2. Currency Formatting Preview Helper
    // ---------------------------------------------------------
    const currencyInputs = [
        'income_annum',
        'loan_amount',
        'residential_assets_value',
        'commercial_assets_value',
        'luxury_assets_value',
        'bank_asset_value'
    ];

    currencyInputs.forEach(id => {
        const input = document.getElementById(id);
        const preview = document.getElementById(`${id}_preview`);
        if (input && preview) {
            const updatePreview = () => {
                const val = parseFloat(input.value);
                if (!isNaN(val) && val >= 0) {
                    preview.textContent = `≈ ₹ ${val.toLocaleString('en-IN')}`;
                    preview.style.display = 'block';
                } else {
                    preview.textContent = '';
                    preview.style.display = 'none';
                }
            };
            input.addEventListener('input', updatePreview);
            updatePreview(); // initial on load
        }
    });

    // ---------------------------------------------------------
    // 3. Dynamic CIBIL Score Badge & Indicator
    // ---------------------------------------------------------
    const cibilInput = document.getElementById('cibil_score');
    const cibilBadge = document.getElementById('cibil_badge');

    if (cibilInput && cibilBadge) {
        const updateCibil = () => {
            const score = parseInt(cibilInput.value);
            cibilBadge.className = 'cibil-badge';

            if (isNaN(score) || score < 300 || score > 900) {
                cibilBadge.textContent = 'Enter 300 - 900';
                cibilBadge.classList.add('cibil-poor');
                return;
            }

            if (score >= 750) {
                cibilBadge.textContent = `${score} • Excellent`;
                cibilBadge.classList.add('cibil-excellent');
            } else if (score >= 650) {
                cibilBadge.textContent = `${score} • Good`;
                cibilBadge.classList.add('cibil-good');
            } else if (score >= 550) {
                cibilBadge.textContent = `${score} • Fair / Average`;
                cibilBadge.classList.add('cibil-average');
            } else {
                cibilBadge.textContent = `${score} • Poor Credit`;
                cibilBadge.classList.add('cibil-poor');
            }
        };

        cibilInput.addEventListener('input', updateCibil);
        updateCibil();
    }

    // ---------------------------------------------------------
    // 4. Test Case Preset Injection (Realistic test cases)
    // ---------------------------------------------------------
    window.loadTestCase = (caseNumber) => {
        const presets = {
            1: {
                // Strong Applicant -> Expected: Approved (~90.8% Probability)
                no_of_dependents: 0,
                education: 'Not Graduate',
                self_employed: 'No',
                income_annum: 3800000,
                loan_amount: 13300000,
                loan_term: 6,
                cibil_score: 655,
                residential_assets_value: 9600000,
                commercial_assets_value: 0,
                luxury_assets_value: 8600000,
                bank_asset_value: 2600000
            },
            2: {
                // Weak Applicant -> Expected: Rejected (~4.8% Approval, 95.2% Rejection)
                no_of_dependents: 5,
                education: 'Not Graduate',
                self_employed: 'No',
                income_annum: 8300000,
                loan_amount: 17700000,
                loan_term: 12,
                cibil_score: 334,
                residential_assets_value: 19600000,
                commercial_assets_value: 13200000,
                luxury_assets_value: 22500000,
                bank_asset_value: 9800000
            },
            3: {
                // Mixed / Borderline Applicant -> Expected: Approved (~62.1% Probability)
                no_of_dependents: 2,
                education: 'Not Graduate',
                self_employed: 'No',
                income_annum: 3200000,
                loan_amount: 8600000,
                loan_term: 18,
                cibil_score: 638,
                residential_assets_value: 1000000,
                commercial_assets_value: 4900000,
                luxury_assets_value: 8600000,
                bank_asset_value: 3500000
            }
        };

        const data = presets[caseNumber];
        if (!data) return;

        for (const [key, val] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = val;
                // Dispatch input event to update previews and badges
                el.dispatchEvent(new Event('input'));
            }
        }

        // Show brief visual feedback on preset applied
        const toastNotice = document.getElementById('preset-notice');
        if (toastNotice) {
            toastNotice.textContent = `Applied Test Case ${caseNumber} preset data!`;
            toastNotice.style.display = 'block';
            setTimeout(() => {
                toastNotice.style.display = 'none';
            }, 3000);
        }
    };

    window.resetForm = () => {
        const form = document.getElementById('prediction-form');
        if (form) {
            form.reset();
            // Trigger input events to refresh badges
            document.querySelectorAll('input').forEach(i => i.dispatchEvent(new Event('input')));
        }
    };

    // ---------------------------------------------------------
    // 5. Auto-Dismiss Flash Alerts
    // ---------------------------------------------------------
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert-dismissible');
        alerts.forEach(alert => {
            try {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                if (bsAlert) bsAlert.close();
            } catch (e) {
                alert.style.display = 'none';
            }
        });
    }, 6000);
});
