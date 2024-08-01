function [outputArg1] = probs(name, savepath)
% This function computes zero and one-dimensional barcodes of a given string quartet 
% in midi file format. It needs Javaplex library.
%
% INPUT midi file of the string quartet or a list name with the list included in the function list.m
%
% OUTPUT .txt files with the barcodes. One file per track and per dimension zero and one. Files have as first entry
% in each row the bird time of a class and as a second entry, the time when it dies.
% In the name file 'filename_i_edges_SP_distmat_intervals_j_right_format.txt', 
% i stands for the track number and j for the dimension of the barcode. 

% Here the used function vietoris_rips_javaplexDM.m is a modified version of vietoris_rips_javaplex.m written by
% Nina Otter for  A roadmap for the computation of persistent homology. N. Otter, M. Porter, U. Tillmann, 
% P. Grindrod, H. Harrington, EPJ Data Science 2017
outputArg1 = 1;
 if size(strfind(name, '.mid'))~=0
  ph(name);
  fprintf("Found it", name);
 else 
  fprintf("There is no list of works named %s or the .mid extension was missed", name);
  outputArg1 = 2;
 end

 function []=ph(midiname)
  %  As we are working with string quartets, we extract the information 
  %  from the four instruments and storage in different matrices. In the process we convert polyphonic tracks 
  %  into monophonic in order to be able to apply the pcdist2 function from MIDI Toolbox. 
  nmat=readmidi(midiname);
  i=1;
  while i<size(nmat,1) && nmat(i,1)<=nmat(i+1,1)
   i=i+1;
  end
  nm1=nmat(1:i,:);
  nm1=extreme(nm1,'high');
  i2=i+1
  i=i+1;
  while i<size(nmat,1) && nmat(i,1)<=nmat(i+1,1)
   i=i+1;
  end
  nm2=nmat(i2:i,:);
  nm2=extreme(nm2,'high');
  i3=i+1
  i=i+1;
  while i<size(nmat,1) && nmat(i,1)<=nmat(i+1,1)
   i=i+1;
  end
  nm3=nmat(i3:i,:);
  nm3=extreme(nm3,'high');
  i4=i+1
  i=i+1;
  while i<size(nmat,1) && nmat(i,1)<=nmat(i+1,1)
   i=i+1;
  end
  nm4=nmat(i4:i,:);
  nm4=extreme(nm4,'high');

  nmpitch1=pcdist2SC2(nm1);
  nmpitch2=pcdist2SC2(nm2);
  nmpitch3=pcdist2SC2(nm3);
  nmpitch4=pcdist2SC2(nm4);
  filename1 = strcat(savepath,'\1.mat');
  filename2 = strcat(savepath,'\2.mat');
  filename3 = strcat(savepath,'\3.mat');
  filename4 = strcat(savepath,'\4.mat');

  save(filename1,'nmpitch1')
  save(filename2,'nmpitch2')
  save(filename3,'nmpitch3')
  save(filename4,'nmpitch4')

 end
end
