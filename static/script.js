// Global variables for pagination
let currentData = [];
let currentPage = 1;
const rowsPerPage = 10;

// Initialize GSAP animation for hero content
gsap.to(".hero-content", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power3.out"
});

// Function to redirect to the prediction form page (if needed)
function redirectToForm() {
    window.location.href = "form.html";
}

// Handle file selection and CSV parsing using PapaParse (if used on other pages)
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        alert("Please select a file!");
        return;
    }

    Papa.parse(file, {
        complete: function(results) {
            if (results.data.length === 0) {
                alert("The CSV file is empty!");
                return;
            }

            // Filter out empty rows
            currentData = results.data.filter(row => Object.values(row).some(value => value.trim() !== ""));
            currentPage = 1;
            displayData();

            document.getElementById('totalRows').textContent = `Total rows: ${currentData.length - 1}`;
            document.getElementById('tableContainer').style.display = 'block';
        },
        header: true
    });
}

// Display dataset in a table with pagination
document.addEventListener("DOMContentLoaded", function () {
    // Load CSV file automatically when the page loads
    fetch('/static/data/dataset.csv')
        .then(response => response.text())
        .then(csvText => {
            Papa.parse(csvText, {
                header: true,
                complete: function (results) {
                    if (results.data.length === 0) {
                        alert("The CSV file is empty!");
                        return;
                    }

                    // Filter out empty rows
                    currentData = results.data.filter(row => Object.values(row).some(value => value.trim() !== ""));
                    currentPage = 1;
                    displayData();

                    document.getElementById('totalRows').textContent = `Total rows: ${currentData.length}`;
                    document.getElementById('tableContainer').style.display = 'block';
                }
            });
        })
        .catch(error => console.error("Error loading dataset:", error));
});


    // Update pagination info
    document.getElementById('pageInfo').textContent = 
        `Page ${currentPage} of ${Math.ceil(currentData.length / rowsPerPage)}`;
}

// Pagination controls
function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        displayData();
    }
}

function nextPage() {
    if (currentPage < Math.ceil(currentData.length / rowsPerPage)) {
        currentPage++;
        displayData();
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Animate sections on scroll
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        const rect = section.getBoundingClientRect();
        if (rect.top < window.innerHeight * 0.75) {
            section.style.opacity = 1;
            section.style.transform = 'translateY(0)';
        }
    });
});

