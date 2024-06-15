const expressions = ['kiss', 'relaxed', 'smile', 'surprised', 'wink'];

function updateImage() {
    const imageIndex = document.getElementById('imageSlider').value;
    const identity = document.getElementById('identitySlider').value;
    const expressionIndex = document.getElementById('expressionSlider').value;
    const expression = expressions[expressionIndex];
    
    const imagePath = `./results/${identity}/${expression}/${imageIndex}.png`;
    document.getElementById('displayImage').src = imagePath;
    document.getElementById('expressionName').textContent = expression;
    document.getElementById('identityName').textContent = `stylegan${identity}`;
    document.getElementById('identityImage').src = `./stylegan_ids/stylegan${identity}.png`;
    document.getElementById('expressionImage').src = `./demo_exps/${expression}.png`;
}

document.getElementById('imageSlider').addEventListener('input', updateImage);
document.getElementById('identitySlider').addEventListener('input', updateImage);
document.getElementById('expressionSlider').addEventListener('input', updateImage);

// Initial update to set the correct image when the page loads
updateImage();
