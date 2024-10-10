const photoData = [
  // More
  { label: "JPG", src: "images/file_type/jpg.jpg" },
  { label: "PNG", src: "images/file_type/png.png" },

  // Size Test
  { label: "Size_1000", src: "images/size_test/1000.png" },
  { label: "Size_800", src: "images/size_test/800.png" },
  { label: "Size_600", src: "images/size_test/600.png" },
  { label: "Size_400", src: "images/size_test/400.png" },
  { label: "Size_200", src: "images/size_test/200.png" },
  { label: "Size_100", src: "images/size_test/100.png" },
  // Color Test
  { label: "L100", src: "images/color_test/L100.png" },
  { label: "L60", src: "images/color_test/L60.png" },
  { label: "L20", src: "images/color_test/L20.png" },
  { label: "L10", src: "images/color_test/L10.png" },
  { label: "L5", src: "images/color_test/L5.png" },
  { label: "L1", src: "images/color_test/L1.png" },
  // Blurness
  { label: "blur1", src: "images/blurness/blur1.png" },
  { label: "blur2", src: "images/blurness/blur2.png" },
  { label: "blur5", src: "images/blurness/blur5.png" },
  { label: "blur10", src: "images/blurness/blur10.png" },
  { label: "blur15", src: "images/blurness/blur15.png" },
  { label: "blur20", src: "images/blurness/blur20.png" },
  // Resolution
  { label: "16_9", src: "images/resolution/16_9.png" },
  { label: "9_16", src: "images/resolution/9_16.png" },
  { label: "4_3", src: "images/resolution/4_3.png" },
  { label: "3_4", src: "images/resolution/3_4.png" },
  { label: "3_2", src: "images/resolution/3_2.png" },
  { label: "2_3", src: "images/resolution/2_3.png" },
];

window.addEventListener("DOMContentLoaded", function () {
  const photoLibraryDiv = document.querySelector(".photo-library");

  photoData.forEach((photo) => {
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
