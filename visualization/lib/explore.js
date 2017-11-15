/*
#    Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of mhan.
#
#    mhan is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    mhan is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with mhan. If not, see http://www.gnu.org/licenses/
*/

var request = new XMLHttpRequest();
var langs = ['english', 'german','spanish','portuguese','ukrainian','russian','arabic','persian'];
var cur_lang = 'english';
var cur_id = 0;
var fnames = {
    'multikw-en': 'MHAN: en-de → en',
    'multikw-de': 'MHAN: en-de → de'
      };

function load(ltype)
{
   $('#selected_type').html(fnames[ltype]);
   window.cat = {};
   $.each(langs, function(lid, lang){
       try{
	   request.open("GET", ltype+"_"+lang+".json", false);
       	   request.send(null);
       	   window.cat[lang] = JSON.parse(request.responseText);
   	}catch(err){}
   });
   lid = langs.indexOf(cur_lang);
   if (lid < Object.keys(window.cat).length)
   	  cur_lang = Object.keys(window.cat)[lid];
   else
	    cur_lang = Object.keys(window.cat)[0];
   $('#'+cur_lang+'_radio').click();
   list_reviews();
}

load(Object.keys(fnames)[0]);

function list_reviews()
{
  $('#group_1').html('');
  var group = 1;
  $.each(Object.keys(cat[cur_lang]), function(index, review){
     review = review.split('.json')[0]
     var active = '';
     if(index == 0)
     {
       active = "active";
     }
     id = review[0]['idx']
     $('#group_'+group).append('<a onclick="list_reviews();load_doc('+index+');"class="list-group-item list-group-item-action" title="index/file" id="rev'+index+'">'+index+'/'+review+'</a>')
  });
}


function load_doc(id)
{
    show_review(id, cur_lang)
}

function show_tags(language, doc)
{
  tags = doc['tags'];
  $('#'+language).html('');
  $.each(tags, function(tag_idx, tag){
    tagname = tag[0];
    tagconf = tag[1]*5;
    $('#'+language).append('<div class="tag" style="opacity:'+tagconf+'" >'+tagname+' ('+(tagconf/5).toFixed(3)+')</div>');
    if (tag_idx == 4)
      return false
  });
   $("#"+language+"_radio-div").show();

}

function show_options()
{
	$.each(Object.keys(fnames), function(idx, x){
	 	$('#dropdown').append("<li id='"+x+"'><a href='#'>"+fnames[x]+"</a></li>")
		$('#'+x ).click(function() {
			 load(x);
			});
		});
}

function activate(lang)
{
  cur_lang = lang;
  test = "#"+lang+"_radio";
  show_review(cur_id, lang);
}

function show_review(id, lang)
{
  cur_id = id;
  keys = Object.keys(window.cat[lang]);
  doc = window.cat[lang][keys[id]];
  $('.temp').remove();
  $('#rev'+id).addClass('active');
  texts = doc['text'];
  satts = doc['satts'];
  watts = doc['watts'];

  $.each(langs, function(lid, clang){
    try{
	    show_tags(clang, window.cat[clang][keys[id]]);
 	}catch(err)
	{
	    $('#'+clang).html("");
 	    $("#"+clang+"_radio-div").hide();
	}
  });

  $.each(texts, function(index, sentence){
      ove_score = parseFloat(satts[index]).toFixed(3);
      var text =  '';
      if (typeof watts[index] != 'undefined')
      {
	      $.each(sentence, function(jindex, word){
	    	  var color = " 0, 210, 252 ";
          if (typeof watts[index][jindex] == 'undefined')
            wove_score = 0;
          else
          	wove_score = parseFloat(watts[index][jindex]).toFixed(3)*2;
          text += '<span title="sent_'+index+'/word_'+jindex+': '+wove_score+'" class="words" style=" background-color:rgba('+color+', '+wove_score+');">'+word+'</span>';
		  });
	      color = "252, 31, 1";
	      cur = '<tr class="temp">';
	      cur += '<td class="colorcol" style="background-color: rgba('+color+', '+(ove_score*1.5)+');" >'+ove_score+'</td>';

	      cur += '<td class="sentcol" style="word-wrap: break-word"> <div style="padding:3px;width:">'+text+'</div></td>';
	      cur += '</tr>';
	      $('#current_table tr:last').after(cur);
        }
     });

}

function show_contents(talk_id)
{
   $.each(langs, function(lid, lang){
	if( typeof window.cat[lang] == "undefined")
		$('#'+lang+'_radio-div').hide()
   });
   show_options();
   $('#content').hide();
   $('#footer').hide();
   $('#loading').show();
   list_reviews();
   $('#'+cur_lang+'_radio').click();
   setTimeout(function() {
     $('#loading').hide();
     $('#content').show();
     $('#footer').show();
   }, 300);
}


$(window).load(function ()
{
     show_contents();
     $('#selected_type').html(fnames[Object.keys(fnames)[0]]);

});
