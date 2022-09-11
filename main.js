$(function() {

  ws = new WebSocket('ws://98.238.221.173:8765')

  ws.onopen = function() {
    ws.send(JSON.stringify('Connected'));
    $('#submit').html('Generate!')
  };

  ws.onclose = function() {
    $('#submit').css('filter', 'saturate(1.5) hue-rotate(100deg)')
      .html('Disconnected').prop('disabled', true);
  }

  var prompt = ''
  ws.onmessage = function(event) {
    packet = JSON.parse(event.data);
    if (packet == 'Done') {
      $('#submit').css('filter', 'saturate(1.5) hue-rotate(280deg)')
        .html('Finished!').prop('disabled', false);
    }
    else if (typeof packet == 'number') {
      if (packet > 2) {
        $('#warning').html(`${packet} clients connected, expect slowdown!`).fadeIn(250);
      }
      else { $('#warning').fadeOut(250); }
    }
    else {
      $('#submit').html('Generating...');
      prompt = prompt.concat(packet);
      $('#result').val(prompt);
      $('#result:not(:hover)').scrollTop($('#result')[0].scrollHeight);
    };
  };

  $('#submit:enabled').on('click', function() {
    prompt = '';
    prompt = $('#prompt').val();

    ws.send(JSON.stringify([$('#chars').val(), prompt]))
    $(this).css('filter', '').html('Loading...').prop('disabled', true);
  })

  $('#chars').css('width', `${$('#chars')[0].scrollWidth}px`);

  $('#chars').on('input', function() {
    $(this).css('width', '');
    $(this).css('width', `${$(this)[0].scrollWidth}px`);
  })
})
