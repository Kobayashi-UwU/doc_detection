const photoData = [
    { label: 'L100', src: 'images/test1.png' },
    { label: 'L80', src: 'images/test2.jpg' },
    // Add more images here
  ];
  
  window.addEventListener("DOMContentLoaded", function () {
    const photoLibraryDiv = document.querySelector(".photo-library");
  
    photoData.forEach(photo => {
      const photoItem = document.createElement("div");
      photoItem.classList.add("photo-item");
  
      const label = document.createElement("p");
      label.innerText = photo.label;
      photoItem.appendChild(label);
  
      const img = document.createElement("img");
      img.src = photo.src;
      img.alt = `Test Image ${photo.label}`;
      img.width = 100;
      img.height = 100;
      img.classList.add("library-photo");
  
      photoItem.appendChild(img);
      photoLibraryDiv.appendChild(photoItem);
    });
  });
  