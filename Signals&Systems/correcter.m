clear
clc

answer_column='c5:c13';

write_mark_column='aq';

student_ID_column='d2:d50';

filenamewriter='ListStudents.xlsx';

Correctanswers=xlsread('Corrects.xlsx',1,answer_column);

studentIDvec=xlsread(filenamewriter,1,student_ID_column);

num_of_answers=length(Correctanswers);

for ind=1:50
    
%     try
        filename=['1 (' num2str(ind) ').xlsx'];

        Studentanswers=xlsread(filename,1,answer_column);
        
        Studentanswers=[Studentanswers;zeros(num_of_answers-length(Studentanswers),1)];

%         disp([Studentanswers,Correctanswers])
        
%         disp(Correctanswers)

        studentID=xlsread(filename,1,'f7:f7');

        StudentMark=ceil(mean(Studentanswers==Correctanswers)*100);

        studentInd=find(studentIDvec==studentID);
        
        disp(StudentMark)
%         disp(studentInd)

        disp(studentID)

    %     disp(StudentMark)

        xlswrite(filenamewriter,StudentMark,1,[write_mark_column num2str(studentInd+1)])
        
        disp([num2str(ind) ' Successful!'])
    % disp(['ag' num2str(studentInd+1)])
%         writetable(StudentMark,filenamewriter,'Sheet',1,'Range',['ag' num2str(studentInd+1)])
        
%         error('dkfl;dsf')
%     catch Error
% %         disp(Error)
%         disp([Error.identifier num2str(studentID)])
%         disp(ind)
%         error('1')
%     disp(ind)
%     disp('dlsdsldjsjdk')
    
%     disp(studnetID)
% disp('djfkldsjfsldkf')    
% disp(ind)
%     disp(StudentMark)
%     end
%     disp(Studentanswers-Correctanswers)
end