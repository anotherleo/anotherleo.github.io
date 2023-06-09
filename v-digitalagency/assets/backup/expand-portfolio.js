const expandControls = document.querySelector('.expand__controls');
const expandButton = document.querySelector('.button__expand');
const expandLeft = document.querySelector('.expanded__left');

expandButton.addEventListener('click', () => {
    expandLeft.classList.toggle('is--open');
    expandControls.classList.toggle('is--open');

    if (expandControls.classList.contains('is--open')) {
        expandButton.textContent = 'Reduzir';
    } else {
        expandButton.textContent = 'Expandir Portfolio';
    }
});