function uploadAndClassifyImage1() {
    var fileInput = document.getElementById('imageFile1').files;
    if (!fileInput.length) {
        return alert('Please choose an image to upload!');
    }

    var file = fileInput[0];
    var filename = file.name;

    console.log(filename);

    var formData = new FormData();
    formData.append(filename, file);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response){
        var resp = JSON.parse(response);
        console.log(resp);
        document.getElementById('result1').textContent = resp.class;
    })
    .fail(function() {
        alert ("There was an error while sending the prediction request.");
    });
};

function uploadAndClassifyImage2() {
    var fileInput = document.getElementById('imageFile2').files;
    if (!fileInput.length) {
        return alert('Please choose an image to upload!');
    }

    var file = fileInput[0];
    var filename = file.name;

    console.log(filename);

    var formData = new FormData();
    formData.append(filename, file);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response){
        var resp = JSON.parse(response);
        console.log(resp);
        document.getElementById('result2').textContent = resp.class;
    })
    .fail(function() {
        alert ("There was an error while sending the prediction request.");
    });
};

function uploadAndAlignFace() {
    var fileInput = document.getElementById('imageFile3').files;
    if (!fileInput.length) {
        return alert('Please choose an image to upload!');
    }

    var imagebox = $('#imagebox')
    var file = fileInput[0];
    var filename = file.name;

    console.log(filename);

    var formData = new FormData();
    formData.append(filename, file);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response){
        var resp = JSON.parse(response);
        console.log(resp.file);
        var bytestring = resp['aligned'];
        var image = bytestring.split('\'')[1];
        imagebox.attr('src' , 'data:image/jpeg;base64,'+image);
    })
    .fail(function() {
        alert ("There was an error while sending the prediction request.");
    });
};

function uploadAndSwapFace() {
    var fileInput1 = document.getElementById('imageFile4').files;
    if (!fileInput1.length) {
        return alert('Please choose an image to upload!');
    }

    var fileInput2 = document.getElementById('imageFile5').files;
    if (!fileInput2.length) {
        return alert('Please choose an image to upload!');
    }

    var imagebox = $('#imagebox1')

    var file1 = fileInput1[0];
    var filename1 = file1.name;

    console.log(filename1);

    var file2 = fileInput2[0];
    var filename2 = file2.name;

    console.log(filename2);

    var formData = new FormData();
    formData.append(filename1, file1);
    formData.append(filename2, file2);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response){
        var resp = JSON.parse(response);
        console.log(resp.file);
        var bytestring = resp['swapped'];
        var image = bytestring.split('\'')[1];
        imagebox.attr('src' , 'data:image/jpeg;base64,'+image);
    })
    .fail(function() {
        alert ("There was an error while sending the prediction request.");
    });
};
